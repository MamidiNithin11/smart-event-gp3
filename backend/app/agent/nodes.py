from __future__ import annotations
import json
import logging
import math
import re
import time
from datetime import date
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.agent import memory
from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.core.ai_config import ai_settings
from app.core.permissions import AuthorizationError, GUEST_ROLE
from app.llm.exceptions import LLMProviderError, safe_error_category
from app.llm.router import LLMProviderRouter
from app.observability.logger import log_tool_call
from app.observability.timing import emit_timing
from app.tools.context import PermissionDenied, get_tool_context
from app.tools.formatting import parse_datetime
from app.tools.registry import TOOLS_BY_NAME, get_tools_for_role
logger = logging.getLogger(__name__)
_SENSITIVE_TOOL_ARGUMENTS = frozenset({'password', 'secret', 'token', 'api_key', 'authorization'})
_REGISTRATION_COUNT_PATTERNS = (
    re.compile(
        r'\bhow\s+many\s+(?:(?:people|persons|attendees|participants|registrations|registrants)\s+)?'
        r'(?:have\s+)?(?:registered|register|signed\s+up|booked|enrolled)\s+'
        r'(?:for|in)\s+(?P<event>.+?)\s*$',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:number|count)\s+of\s+(?:confirmed\s+)?registrations?\s+'
        r'(?:for|in)\s+(?P<event>.+?)\s*$',
        re.IGNORECASE,
    ),
)

def _registration_count_event_name(message: str) -> str | None:
    for pattern in _REGISTRATION_COUNT_PATTERNS:
        match = pattern.search(message.strip().rstrip('?.!'))
        if match:
            event_name = match.group('event').strip().strip('"\'')
            event_name = re.sub(r'\s+event$', '', event_name, flags=re.IGNORECASE).strip()
            if event_name:
                return event_name
    return None

def _is_general_venue_directory_request(message: str) -> bool:
    if re.search(
        r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|'
        r'today|tomorrow|this\s+(?:morning|afternoon|evening)|'
        r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
        r'january|february|march|april|may|june|july|august|september|'
        r'october|november|december)\b',
        message,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r'\b(?:list|show|give|display)\s+(?:me\s+)?'
            r'(?:(?:all|the)\s+)?(?:available\s+)?venues?\b',
            message,
            re.IGNORECASE,
        )
        or re.search(
            r'\b(?:what|which)\s+(?:are\s+)?(?:(?:all|the)\s+)?'
            r'(?:available\s+)?venues?\b',
            message,
            re.IGNORECASE,
        )
    )

def _redact_tool_arguments(arguments: dict) -> dict:
    return {
        key: '[REDACTED]' if key.lower() in _SENSITIVE_TOOL_ARGUMENTS else value
        for key, value in arguments.items()
    }

def _structured_event_creation_args(message: str) -> dict | None:
    if not re.search(r'\bcreate\s+(?:(?:an?|the)\s+)?event\b', message, re.IGNORECASE):
        return None

    labels = (
        r'Registration\s+deadline|Start\s+time|End\s+time|Date\s+of\s+event|'
        r'Event\s+title|Title|Category|Venue|Date|Capacity|Description|Status'
    )
    matches = list(re.finditer(
        rf'(?<!\w)(?P<label>{labels})(?:\s+is\b)?\s*(?::|=)?\s*',
        message,
        re.IGNORECASE,
    ))
    fields = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(message)
        key = re.sub(r'\s+', ' ', match.group('label').lower())
        if key == 'event title':
            key = 'title'
        elif key == 'date of event':
            key = 'date'
        value = message[match.end():end].strip().strip('*').strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1].strip()
        fields[key] = value

    required = ('title', 'category', 'venue', 'date', 'start time', 'end time', 'capacity')
    if any(not fields.get(field) for field in required):
        return None
    if fields.get('status', 'draft').casefold() != 'draft':
        return None
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', fields['date']):
        return None
    if not re.fullmatch(r'\d{1,2}:\d{2}', fields['start time']):
        return None
    if not fields['capacity'].isdigit() or int(fields['capacity']) <= 0:
        return None

    try:
        event_date = f"{fields['date']}T{fields['start time']}"
        parsed_start = parse_datetime(event_date)
        end_value = fields['end time']
        if re.fullmatch(r'\d{1,2}:\d{2}', end_value):
            end_value = f"{fields['date']}T{end_value}"
        elif not re.fullmatch(r'\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?', end_value):
            return None
        end_time = parse_datetime(end_value)
        if end_time <= parsed_start:
            return None
        deadline = fields.get('registration deadline', '')
        if deadline:
            parse_datetime(deadline)
    except ValueError:
        return None

    return {
        'title': fields['title'],
        'category': fields['category'],
        'event_date': event_date,
        'end_time': end_value,
        'capacity': int(fields['capacity']),
        'description': fields.get('description', ''),
        'registration_deadline': fields.get('registration deadline', ''),
    }

def _has_successful_availability_check(state: AgentState) -> bool:
    results = state.get('tool_results', [])
    calls = state.get('tool_calls', [])
    if not results or not calls:
        return False
    result = results[-1]
    if (
        result.get('name') != 'check_venue_availability'
        or result.get('status') != 'success'
        or not str(result.get('output_preview', '')).startswith('AVAILABLE:')
    ):
        return False
    return any(call.get('name') == 'check_venue_availability' for call in reversed(calls))

def load_memory(state: AgentState) -> dict:
    context = get_tool_context()
    started = time.perf_counter()
    history = memory.load_history(db=context.db, session_id=state['session_id'], limit=ai_settings.MEMORY_WINDOW_MESSAGES)
    history_ms = (time.perf_counter() - started) * 1000
    emit_timing('db.conversation_history', history_ms, history_messages=len(history))
    prompt_started = time.perf_counter()
    messages = [SystemMessage(content=build_system_prompt(role=context.role, today=date.today().strftime('%A, %d %B %Y'))), *history, HumanMessage(content=state['message'])]
    prompt_ms = (time.perf_counter() - prompt_started) * 1000
    tools = get_tools_for_role(context.role)
    message_chars = sum(len(str(message.content)) for message in messages)
    tool_chars = sum(
        len(tool.name) + len(tool.description or '') + len(str(tool.args))
        for tool in tools
    )
    emit_timing(
        'prompt.profile',
        prompt_ms,
        message_count=len(messages),
        history_messages=len(history),
        tool_count=len(tools),
        message_chars=message_chars,
        tool_schema_chars=tool_chars,
        estimated_input_tokens=math.ceil((message_chars + tool_chars) / 4),
        token_estimate='characters_divided_by_4',
    )
    emit_timing('langgraph.load_memory', history_ms + prompt_ms)
    return {'messages': messages}

def after_load_memory(state: AgentState) -> str:
    context = get_tool_context()
    if context.is_admin and _registration_count_event_name(state['message']):
        return 'registration_count'
    if _is_general_venue_directory_request(state['message']):
        return 'venue_directory'
    return 'reason'

def fast_registration_count(state: AgentState) -> dict:
    started = time.perf_counter()
    event_name = _registration_count_event_name(state['message'])
    if not event_name:
        emit_timing(
            'langgraph.registration_count',
            (time.perf_counter() - started) * 1000,
            outcome='event_name_missing',
        )
        return {'error': 'Could not identify the event name for the registration count.'}

    tool_request = AIMessage(
        content='',
        tool_calls=[{
            'name': 'get_event_registration_count',
            'args': {'event_name': event_name},
            'id': 'fast-registration-count',
        }],
    )
    result = act({**state, 'messages': [*state['messages'], tool_request]})
    emit_timing(
        'langgraph.registration_count',
        (time.perf_counter() - started) * 1000,
        outcome='success',
    )
    return {
        **result,
        'intent': 'registration_count',
    }

def fast_venue_directory(state: AgentState) -> dict:
    tool_request = AIMessage(
        content='',
        tool_calls=[{
            'name': 'search_venues',
            'args': {'name': ''},
            'id': f'venue-directory-{state["run_id"]}',
        }],
    )
    result = act({**state, 'messages': [*state['messages'], tool_request]})
    tool_messages = result.get('messages', [])
    if tool_messages:
        result['final_response'] = str(tool_messages[-1].content).strip()
    result['intent'] = 'venue_directory'
    return result

def reason(state: AgentState) -> dict:
    started = time.perf_counter()
    role = get_tool_context().role
    tools = get_tools_for_role(role)
    logger.info('reason(): routing agent request through primary provider chain for role=%s', role)
    try:
        router = LLMProviderRouter()
        response = router.invoke(
            state['messages'],
            temperature=ai_settings.CHAT_TEMPERATURE,
            tools=tools,
            operation='agent_reason',
        )
    except LLMProviderError as exc:
        emit_timing('langgraph.reason', (time.perf_counter() - started) * 1000, failed=True)
        logger.warning('Configured chat providers unavailable during agent reasoning.')
        provider_error = getattr(exc, 'cause', None)
        if safe_error_category(provider_error or exc) == 'authentication_failed':
            reply = (
                "I can't generate an AI response right now because the configured AI provider rejected "
                'its API key. Please ask the site administrator to verify the Groq and OpenRouter API '
                'keys; once corrected, you can try again.'
            )
        else:
            reply = (
                "I'm unable to reach the AI service right now. Please try again later, or contact the "
                'site administrator if the problem continues.'
            )
        return {
            'messages': [AIMessage(content=reply)],
            'error': 'Configured chat providers unavailable.',
        }
    emit_timing('langgraph.reason', (time.perf_counter() - started) * 1000, failed=False)
    return {'messages': [response], 'loop_count': state.get('loop_count', 0) + 1}

def act(state: AgentState) -> dict:
    node_started = time.perf_counter()
    context = get_tool_context()
    last_message = state['messages'][-1]
    requested = getattr(last_message, 'tool_calls', None) or []
    tool_messages: list[ToolMessage] = []
    call_records: list[dict] = []
    result_records: list[dict] = []
    citations: list[str] = []
    direct_response = ''
    allowed_tool_names = {allowed_tool.name for allowed_tool in get_tools_for_role(context.role)}
    for sequence, call in enumerate(requested):
        name = call.get('name', '')
        args = call.get('args', {}) or {}
        logged_args = _redact_tool_arguments(args)
        call_id = call.get('id', '')
        started = time.perf_counter()
        status = 'success'
        error = None
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            output = f"There is no tool called '{name}'. Available tools: {', '.join(sorted(TOOLS_BY_NAME))}."
            status = 'error'
            error = 'unknown tool'
        elif name not in allowed_tool_names:
            output = 'Sign in to use this tool.' if context.role == GUEST_ROLE else 'Only an administrator can use this tool.'
            status = 'denied'
            error = 'role not permitted'
        else:
            try:
                output = tool.invoke(args)
            except PermissionDenied as exc:
                output = str(exc)
                status = 'denied'
                error = str(exc)
            except AuthorizationError as exc:
                output = str(exc)
                status = 'denied'
                error = str(exc)
            except ValueError as exc:
                output = f'Invalid input: {exc}'
                status = 'error'
                error = str(exc)
            except Exception as exc:
                context.db.rollback()
                logger.exception('Tool %s raised', name)
                output = f'The {name} tool failed unexpectedly. Tell the user and do not retry.'
                status = 'error'
                error = f'{exc.__class__.__name__}: {exc}'
        latency_ms = (time.perf_counter() - started) * 1000
        output = str(output)
        if name == 'get_event_registration_count' and status == 'success' and len(requested) == 1:
            direct_response = output
        tool_messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
        call_records.append({'name': name, 'args': logged_args})
        result_records.append({'name': name, 'status': status, 'latency_ms': round(latency_ms, 2), 'output_preview': output[:300]})
        if name == 'search_event_policy' and 'Sources:' in output:
            citations.extend((part.strip() for part in output.split('Sources:', 1)[1].split('\n')[0].split(',') if part.strip()))
        log_started = time.perf_counter()
        log_tool_call(db=context.db, run_id=state['run_id'], tool_name=name, tool_input=logged_args, tool_output=output, latency_ms=latency_ms, status=status, error=error, sequence=sequence)
        emit_timing('db.tool_call_persist', (time.perf_counter() - log_started) * 1000, tool=name)
        emit_timing('tool.execute', latency_ms, tool=name, status=status)
        logger.info('tool=%s status=%s latency=%.0fms', name, status, latency_ms)
    result = {'messages': tool_messages, 'tool_calls': state.get('tool_calls', []) + call_records, 'tool_results': state.get('tool_results', []) + result_records, 'retrieved_documents': state.get('retrieved_documents', []) + citations}
    if direct_response:
        result['final_response'] = direct_response
    emit_timing('langgraph.act', (time.perf_counter() - node_started) * 1000, tool_count=len(requested))
    return result

def finalize(state: AgentState) -> dict:
    node_started = time.perf_counter()
    context = get_tool_context()
    answer = state.get('final_response', '')
    if not answer:
        for message in reversed(state['messages']):
            if isinstance(message, AIMessage) and message.content:
                answer = str(message.content).strip()
                break
    if not answer:
        latest_tool_message = next(
            (
                message
                for message in reversed(state['messages'])
                if isinstance(message, ToolMessage) and message.content
            ),
            None,
        )
        if latest_tool_message is not None:
            answer = str(latest_tool_message.content).strip()
            logger.warning(
                'Agent run %s had no final assistant response; returning the latest tool result',
                state.get('run_id'),
            )
        else:
            answer = (
                "I didn't receive a usable response from the AI model. "
                'Please try again.'
            )
            logger.warning(
                'Agent run %s finished without an assistant response or tool result',
                state.get('run_id'),
            )
    try:
        started = time.perf_counter()
        memory.save_turn(db=context.db, session_id=state['session_id'], user_message=state['message'], assistant_message=answer)
        emit_timing('db.save_conversation_turn_flush', (time.perf_counter() - started) * 1000)
    except Exception as exc:
        context.db.rollback()
        logger.warning('Could not save conversation turn: %s', exc)
    emit_timing('langgraph.finalize', (time.perf_counter() - node_started) * 1000)
    return {'final_response': answer}

def create_structured_event(state: AgentState) -> dict:
    context = get_tool_context()
    details = _structured_event_creation_args(state['message'])
    if not context.is_admin or details is None or not _has_successful_availability_check(state):
        return {}

    availability_call = next(
        call
        for call in reversed(state.get('tool_calls', []))
        if call.get('name') == 'check_venue_availability'
    )
    availability_args = availability_call.get('args', {})
    venue_id = availability_args.get('venue_id')
    if not venue_id:
        return {}
    requested_time = availability_args.get('requested_time')
    if requested_time:
        timezone_hint = parse_datetime(requested_time).tzinfo
        for field in ('event_date', 'end_time', 'registration_deadline'):
            value = details.get(field)
            if value and timezone_hint is not None and parse_datetime(value).tzinfo is None:
                details[field] = parse_datetime(value).replace(tzinfo=timezone_hint).isoformat()

    request = AIMessage(
        content='',
        tool_calls=[{
            'name': 'create_event',
            'args': {**details, 'venue_id': venue_id},
            'id': f'structured-create-{state["run_id"]}',
        }],
    )
    result = act({**state, 'messages': [*state['messages'], request]})
    tool_messages = result.get('messages', [])
    if tool_messages and tool_messages[-1].content:
        result['final_response'] = str(tool_messages[-1].content).strip()
    return result

def should_continue(state: AgentState) -> str:
    if state.get('final_response'):
        return 'finalize'
    if state.get('loop_count', 0) >= ai_settings.AGENT_MAX_TOOL_LOOPS:
        logger.warning('Loop cap of %d reached; forcing finalize', ai_settings.AGENT_MAX_TOOL_LOOPS)
        return 'finalize'
    last_message = state['messages'][-1]
    if getattr(last_message, 'tool_calls', None):
        return 'act'
    return 'finalize'

def after_act(state: AgentState) -> str:
    if state.get('final_response'):
        return 'finalize'
    if (
        _structured_event_creation_args(state.get('message', '')) is None
        or not _has_successful_availability_check(state)
        or any(call.get('name') == 'create_event' for call in state.get('tool_calls', []))
    ):
        return 'reason'
    context = get_tool_context()
    if context.is_admin:
        return 'structured_event_create'
    return 'reason'