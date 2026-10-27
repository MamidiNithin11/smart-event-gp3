from __future__ import annotations
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.agent import nodes as agent_nodes
from app.agent import memory
from app.agent.graph import build_graph, run_agent
from app.agent.nodes import (
    _redact_tool_arguments,
    _is_general_venue_directory_request,
    _registration_count_event_name,
    _structured_event_creation_args,
    act,
    after_act,
    after_load_memory,
    create_structured_event,
    fast_registration_count,
    fast_venue_directory,
    finalize,
    load_memory,
    reason,
    should_continue,
)
from app.agent.prompts import build_system_prompt
from app.agent.state import new_state
from app.core.ai_config import ai_settings
from app.models.agent import ChatMessage
from tests.ai.conftest import FakeSession

def _state(**overrides):
    state = new_state(user_id=uuid.uuid4(), session_id=uuid.uuid4(), run_id=uuid.uuid4(), message='find an AI workshop')
    state.update(overrides)
    return state

def test_graph_compiles_with_the_expected_nodes():
    nodes = set(build_graph().get_graph().nodes)
    for expected in ('load_memory', 'registration_count', 'venue_directory', 'reason', 'act', 'structured_event_create', 'finalize'):
        assert expected in nodes
    assert 'classify_intent' not in nodes

@pytest.mark.parametrize(
    ('message', 'event_name'),
    [
        ('How many people register for ByteBloom event?', 'ByteBloom'),
        ('How many people registered for "ByteBloom"?', 'ByteBloom'),
        ('Number of confirmed registrations for ByteBloom event', 'ByteBloom'),
        ('What is the ByteBloom event about?', None),
    ],
)
def test_registration_count_question_parser(message, event_name):
    assert _registration_count_event_name(message) == event_name

def test_admin_registration_count_query_skips_model(user_context, monkeypatch):
    context = user_context
    context.role = 'ADMIN'
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: context)
    state = _state(message='How many people register for ByteBloom event?')

    assert after_load_memory(state) == 'registration_count'

def test_non_admin_registration_count_question_uses_normal_agent(user_context, monkeypatch):
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: user_context)

    assert after_load_memory(
        _state(message='How many people register for ByteBloom event?')
    ) == 'reason'

@pytest.mark.parametrize(
    'message',
    [
        'give all available venues',
        'List available venues',
        'Show me all venues',
        'What venues are available?',
    ],
)
def test_general_venue_directory_requests_are_routed_without_the_model(message):
    assert _is_general_venue_directory_request(message)

@pytest.mark.parametrize(
    'message',
    [
        'Check which venues are available on 2026-11-02 at 14:00',
        'Show venues available tomorrow',
    ],
)
def test_dated_venue_availability_is_not_a_directory_request(message):
    assert not _is_general_venue_directory_request(message)

def test_general_venue_directory_request_skips_model(user_context, monkeypatch):
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: user_context)
    assert after_load_memory(_state(message='give all available venues')) == 'venue_directory'

def test_fast_venue_directory_returns_the_venue_tool_result(user_context, monkeypatch):
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: user_context)
    venue_tool = MagicMock()
    venue_tool.name = 'search_venues'
    venue_tool.invoke.return_value = 'Found 16 venue(s):\\n- Venue 1\\n- Venue 16'
    monkeypatch.setitem(agent_nodes.TOOLS_BY_NAME, 'search_venues', venue_tool)
    state = _state(message='give all available venues')

    with patch('app.agent.nodes.log_tool_call'):
        result = fast_venue_directory(state)

    assert result['final_response'] == venue_tool.invoke.return_value
    assert result['intent'] == 'venue_directory'
    venue_tool.invoke.assert_called_once_with({'name': ''})

def test_graph_answers_venue_directory_without_asking_model(user_context, monkeypatch):
    venue_tool = MagicMock()
    venue_tool.name = 'search_venues'
    venue_tool.invoke.return_value = 'Found all venues in the directory.'
    monkeypatch.setitem(agent_nodes.TOOLS_BY_NAME, 'search_venues', venue_tool)
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: user_context)
    monkeypatch.setattr('app.agent.memory.load_history', lambda **_kwargs: [])
    state = _state(message='give all available venues')

    with patch('app.agent.nodes.LLMProviderRouter') as router, patch(
        'app.agent.nodes.log_tool_call'
    ), patch('app.agent.nodes.memory.save_turn'):
        result = build_graph().invoke(state)

    assert result['final_response'] == 'Found all venues in the directory.'
    assert result['intent'] == 'venue_directory'
    venue_tool.invoke.assert_called_once_with({'name': ''})
    router.assert_not_called()

def test_fast_registration_count_executes_tool_without_model(admin_context, monkeypatch, caplog):
    import json
    from app.services import event_service, registration_service

    event = type('EventStub', (), {'title': 'ByteBloom', 'event_id': uuid.uuid4()})()
    monkeypatch.setattr(
        event_service,
        'find_events_by_title',
        lambda *_args, **_kwargs: [event],
    )
    monkeypatch.setattr(registration_service, 'get_event_registration_count', lambda *_args, **_kwargs: 7)
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: admin_context)
    state = _state(message='How many people register for ByteBloom event?')

    with caplog.at_level('INFO', logger='app.chat_timing'), patch(
        'app.agent.nodes.LLMProviderRouter'
    ) as router, patch('app.agent.nodes.log_tool_call'):
        result = fast_registration_count(state)

    router.assert_not_called()
    assert result['final_response'] == "'ByteBloom' has 7 confirmed registration(s)."
    assert result['intent'] == 'registration_count'
    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == 'app.chat_timing'
    ]
    assert any(record['stage'] == 'langgraph.registration_count' for record in records)

def test_state_has_every_field_the_roadmap_requires():
    state = _state()
    for field in ('user_id', 'message', 'intent', 'tool_calls', 'tool_results', 'retrieved_documents', 'final_response'):
        assert field in state

def test_collections_start_empty_not_none():
    state = _state()
    assert state['tool_calls'] == []
    assert state['tool_results'] == []
    assert state['retrieved_documents'] == []
    assert state['loop_count'] == 0

def test_tool_calls_route_to_act():
    message = AIMessage(content='', tool_calls=[{'name': 'search_events', 'args': {}, 'id': '1'}])
    assert should_continue(_state(messages=[message], loop_count=1)) == 'act'

def test_registration_count_tool_response_skips_another_llm_turn():
    assert after_act(
        _state(final_response="'ByteBloom' has 7 confirmed registration(s).")
    ) == 'finalize'
    assert after_act(_state()) == 'reason'

def _structured_create_message():
    return (
        'Create an event with the following details:\n'
        'Title: AI and Cloud Computing Workshop\n'
        'Category: Workshop\n'
        'Venue: Bangalore International Exhibition Centre\n'
        'Date: 2026-11-02\n'
        'Start Time: 14:00\n'
        'End Time: 17:00\n'
        'Capacity: 100\n'
        'Description: A technical workshop on AI and cloud computing.\n'
        'Registration Deadline: 2026-10-31 23:59\n'
        'Status: Draft'
    )

def test_structured_event_request_maps_all_fields():
    details = _structured_event_creation_args(_structured_create_message())

    assert details == {
        'title': 'AI and Cloud Computing Workshop',
        'category': 'Workshop',
        'event_date': '2026-11-02T14:00',
        'end_time': '2026-11-02T17:00',
        'capacity': 100,
        'description': 'A technical workshop on AI and cloud computing.',
        'registration_deadline': '2026-10-31 23:59',
    }

def test_structured_event_request_accepts_flattened_inline_fields():
    message = (
        'Create an event with the following details: '
        'Title: AI and Cloud Computing Workshop '
        'Category: Workshop '
        'Venue: Bharat Mandapam '
        'Date: 2026-11-02 '
        'Start Time: 14:00 '
        'End Time: 17:00 '
        'Capacity: 100 '
        'Description: A technical workshop on AI and cloud computing. '
        'Registration Deadline: 2026-10-31 23:59 '
        'Status: Draft'
    )

    details = _structured_event_creation_args(message)

    assert details == {
        'title': 'AI and Cloud Computing Workshop',
        'category': 'Workshop',
        'event_date': '2026-11-02T14:00',
        'end_time': '2026-11-02T17:00',
        'capacity': 100,
        'description': 'A technical workshop on AI and cloud computing.',
        'registration_deadline': '2026-10-31 23:59',
    }

def test_structured_event_request_accepts_user_wording_and_full_end_datetime():
    message = (
        'Create event title "AI Innovation and Future Technologies" '
        'Venue is "Bharat Mandapam" '
        'Date of event 2026-11-05 '
        'Start time 10:00 '
        'End time 2026-11-05 13:00 '
        'Category is Seminar '
        'Capacity 500 '
        'Description "A seminar exploring AI innovation, generative AI, '
        'emerging technologies, and real-world applications." '
        'Registration deadline 2026-11-03 23:59'
    )

    details = _structured_event_creation_args(message)

    assert details == {
        'title': 'AI Innovation and Future Technologies',
        'category': 'Seminar',
        'event_date': '2026-11-05T10:00',
        'end_time': '2026-11-05 13:00',
        'capacity': 500,
        'description': (
            'A seminar exploring AI innovation, generative AI, emerging '
            'technologies, and real-world applications.'
        ),
        'registration_deadline': '2026-11-03 23:59',
    }

def test_structured_creation_only_runs_after_available_venue_check(admin_context, monkeypatch):
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: admin_context)
    state = _state(
        message=_structured_create_message(),
        tool_calls=[{
            'name': 'check_venue_availability',
            'args': {
                'venue_id': 'venue-id',
                'requested_time': '2026-11-02T14:00:00+05:30',
            },
        }],
        tool_results=[{
            'name': 'check_venue_availability',
            'status': 'success',
            'output_preview': 'AVAILABLE: venue is free.',
        }],
    )

    assert after_act(state) == 'structured_event_create'

def test_structured_event_creation_invokes_create_tool_and_returns_its_result(admin_context, monkeypatch):
    monkeypatch.setattr('app.agent.nodes.get_tool_context', lambda: admin_context)
    create_tool = MagicMock()
    create_tool.name = 'create_event'
    create_tool.invoke.return_value = "Created 'AI and Cloud Computing Workshop' in draft status."
    monkeypatch.setitem(agent_nodes.TOOLS_BY_NAME, 'create_event', create_tool)
    state = _state(
        message=_structured_create_message(),
        messages=[HumanMessage(content=_structured_create_message())],
        tool_calls=[{
            'name': 'check_venue_availability',
            'args': {
                'venue_id': 'venue-id',
                'requested_time': '2026-11-02T14:00:00+05:30',
            },
        }],
        tool_results=[{
            'name': 'check_venue_availability',
            'status': 'success',
            'output_preview': 'AVAILABLE: venue is free.',
        }],
    )

    with patch('app.agent.nodes.log_tool_call'):
        result = create_structured_event(state)

    args = create_tool.invoke.call_args.args[0]
    assert args['title'] == 'AI and Cloud Computing Workshop'
    assert args['venue_id'] == 'venue-id'
    assert args['event_date'] == '2026-11-02T14:00:00+05:30'
    assert args['end_time'] == '2026-11-02T17:00:00+05:30'
    assert args['registration_deadline'] == '2026-10-31T23:59:00+05:30'
    assert args['capacity'] == 100
    assert result['final_response'] == create_tool.invoke.return_value

def test_sensitive_tool_arguments_are_redacted_before_logging():
    assert _redact_tool_arguments({
        'email': 'person@example.com',
        'password': 'do-not-log',
        'api_key': 'secret-key',
    }) == {
        'email': 'person@example.com',
        'password': '[REDACTED]',
        'api_key': '[REDACTED]',
    }

def test_a_plain_answer_routes_to_finalize():
    message = AIMessage(content='Here are three workshops.')
    assert should_continue(_state(messages=[message], loop_count=1)) == 'finalize'

def test_loop_cap_forces_finalize():
    message = AIMessage(content='', tool_calls=[{'name': 'search_events', 'args': {}, 'id': '1'}])
    state = _state(messages=[message], loop_count=ai_settings.AGENT_MAX_TOOL_LOOPS)
    assert should_continue(state) == 'finalize'

def test_reason_uses_provider_defaults_and_preserves_tool_binding(user_context):
    from app.tools.registry import get_tools_for_role
    state = _state(messages=[HumanMessage(content='Find events')])
    state['role'] = 'ADMIN'
    tools = get_tools_for_role('USER')
    response = AIMessage(
        content='',
        tool_calls=[{'name': 'search_events', 'args': {'keyword': 'AI'}, 'id': 'call-1'}],
    )
    with patch('app.agent.nodes.LLMProviderRouter') as router, patch(
        'app.agent.nodes.get_tools_for_role', return_value=tools
    ) as get_tools:
        router.return_value.invoke.return_value = response
        result = reason(state)
    assert result['messages'] == [response]
    assert result['loop_count'] == 1
    get_tools.assert_called_once_with('USER')
    assert router.return_value.invoke.call_args.args == (state['messages'],)
    assert router.return_value.invoke.call_args.kwargs['tools'] == tools
    assert router.return_value.invoke.call_args.kwargs['operation'] == 'agent_reason'
    assert 'model' not in router.return_value.invoke.call_args.kwargs

@pytest.mark.parametrize('message', ['Hi', 'What upcoming events are available?'])
def test_greeting_and_general_question_each_make_one_model_call(user_context, message):
    state = _state(message=message, messages=[HumanMessage(content=message)])
    with patch('app.agent.nodes.LLMProviderRouter') as router:
        router.return_value.invoke.return_value = AIMessage(content='Test response')
        result = reason(state)

    router.return_value.invoke.assert_called_once()
    assert result['messages'][0].content == 'Test response'

def test_prompt_profile_telemetry_reports_estimated_tokens_without_prompt_text(user_context, caplog):
    import json
    from app.observability.timing import bind_chat_run

    state = _state(message='Do not log this prompt')
    with bind_chat_run(state['run_id']), caplog.at_level('INFO', logger='app.chat_timing'):
        load_memory(state)

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == 'app.chat_timing'
    ]
    profile = next(record for record in records if record['stage'] == 'prompt.profile')
    assert profile['message_count'] == 2
    assert profile['tool_count'] > 0
    assert profile['estimated_input_tokens'] > 0
    assert 'Do not log this prompt' not in caplog.text

def test_memory_orders_user_before_assistant_when_turn_timestamps_match():
    timestamp = datetime(2026, 10, 26, tzinfo=timezone.utc)
    assistant = ChatMessage(role='assistant', content='Previous answer', created_at=timestamp)
    user = ChatMessage(role='user', content='Previous question', created_at=timestamp)
    db = FakeSession(rows_for={ChatMessage: [assistant, user]})

    history = memory.load_history(db, uuid.uuid4())

    assert [message.content for message in history] == ['Previous question', 'Previous answer']

def test_reason_returns_controlled_message_when_all_providers_fail(user_context):
    from app.llm.exceptions import LLMUnavailableError
    with patch('app.agent.nodes.LLMProviderRouter') as router:
        router.return_value.invoke.side_effect = LLMUnavailableError(
            'multi-provider', 'All configured LLM providers failed.'
        )
        result = reason(_state(messages=[HumanMessage(content='Hello')]))
    assert result['messages'][0].content.startswith("I'm unable to reach the AI service")
    assert result['error'] == 'Configured chat providers unavailable.'

def test_reason_explains_provider_authentication_failure(user_context):
    from app.llm.exceptions import LLMProviderError, LLMUnavailableError
    auth_error = LLMProviderError('openrouter', 'API key rejected.', status_code=401)
    unavailable = LLMUnavailableError(
        'multi-provider',
        'All configured LLM providers failed.',
        cause=auth_error,
    )
    with patch('app.agent.nodes.LLMProviderRouter') as router:
        router.return_value.invoke.side_effect = unavailable
        result = reason(_state(messages=[HumanMessage(content='Hello')]))

    assert 'rejected its API key' in result['messages'][0].content
    assert 'verify the Groq and OpenRouter API keys' in result['messages'][0].content
    assert result['error'] == 'Configured chat providers unavailable.'

def test_act_keeps_rag_source_citations(monkeypatch):
    rag_tool = MagicMock()
    rag_tool.name = 'search_event_policy'
    rag_tool.invoke.return_value = 'Policies say so.\n\nSources: Refund Policy'
    monkeypatch.setitem(agent_nodes.TOOLS_BY_NAME, 'search_event_policy', rag_tool)
    monkeypatch.setattr('app.agent.nodes.get_tools_for_role', lambda _role: [rag_tool])
    state = _state(messages=[_tool_request('search_event_policy')])
    with patch('app.agent.nodes.get_tool_context') as get_context, patch('app.agent.nodes.log_tool_call'):
        get_context.return_value = MagicMock(role='USER', db=MagicMock())
        result = act(state)
    assert result['messages'][0].content == 'Policies say so.\n\nSources: Refund Policy'
    assert result['retrieved_documents'] == ['Refund Policy']

def _tool_request(name, args=None):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args or {}, 'id': 'call-1'}])

def test_successful_tool_call_is_recorded(user_context):
    state = _state(messages=[_tool_request('my_registrations')])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(state)
    assert len(result['tool_calls']) == 1
    assert result['tool_calls'][0]['name'] == 'my_registrations'
    assert result['tool_results'][0]['status'] == 'success'
    assert isinstance(result['messages'][0], ToolMessage)

def test_registration_count_tool_result_is_final_without_second_llm_call(admin_context, monkeypatch):
    from app.services import event_service, registration_service

    event = type('EventStub', (), {'title': 'ByteBloom', 'event_id': uuid.uuid4()})()
    monkeypatch.setattr(
        event_service,
        'find_events_by_title',
        lambda *_args, **_kwargs: [event],
    )
    monkeypatch.setattr(registration_service, 'get_event_registration_count', lambda *_args, **_kwargs: 7)
    state = _state(messages=[_tool_request(
        'get_event_registration_count',
        {'event_name': 'ByteBloom'},
    )])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(state)

    assert result['final_response'] == "'ByteBloom' has 7 confirmed registration(s)."
    assert after_act({**state, **result}) == 'finalize'

def test_unexpected_tool_failure_rolls_back_the_database_session(user_context, monkeypatch):
    failing_tool = MagicMock()
    failing_tool.name = 'my_registrations'
    failing_tool.invoke.side_effect = RuntimeError('database operation failed')
    monkeypatch.setitem(agent_nodes.TOOLS_BY_NAME, failing_tool.name, failing_tool)
    monkeypatch.setattr('app.agent.nodes.get_tools_for_role', lambda _role: [failing_tool])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(_state(messages=[_tool_request(failing_tool.name)]))
    assert user_context.db.rolled_back
    assert result['tool_results'][0]['status'] == 'error'

def test_unknown_tool_returns_a_message_not_an_exception():
    state = _state(messages=[_tool_request('delete_everything')])
    with patch('app.agent.nodes.log_tool_call'), patch('app.agent.nodes.get_tool_context') as ctx:
        ctx.return_value = MagicMock()
        result = act(state)
    assert result['tool_results'][0]['status'] == 'error'
    assert 'no tool called' in result['messages'][0].content.lower()

def test_permission_denial_is_reported_as_denied(user_context):
    state = _state(role='USER', messages=[_tool_request('cancel_event', {'event_id': str(uuid.uuid4())})])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(state)
    assert result['tool_results'][0]['status'] == 'denied'
    assert 'administrator' in result['messages'][0].content.lower()

def test_bad_uuid_becomes_a_recoverable_message(user_context):
    state = _state(messages=[_tool_request('get_event_details', {'event_id': 'not-a-uuid'})])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(state)
    assert result['tool_results'][0]['status'] == 'error'
    assert 'invalid input' in result['messages'][0].content.lower()

def test_tool_latency_is_measured(user_context):
    state = _state(messages=[_tool_request('my_registrations')])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(state)
    assert result['tool_results'][0]['latency_ms'] >= 0

def test_multiple_tool_calls_in_one_turn_are_all_run(user_context):
    message = AIMessage(content='', tool_calls=[{'name': 'my_registrations', 'args': {}, 'id': 'a'}, {'name': 'search_events', 'args': {'keyword': 'ai'}, 'id': 'b'}])
    with patch('app.agent.nodes.log_tool_call'):
        result = act(_state(messages=[message]))
    assert len(result['tool_calls']) == 2
    assert len(result['messages']) == 2

def test_finalize_takes_the_last_assistant_message(user_context):
    state = _state(messages=[HumanMessage(content='find workshops'), AIMessage(content='', tool_calls=[{'name': 'x', 'args': {}, 'id': '1'}]), ToolMessage(content='Found 2 events', tool_call_id='1', name='x'), AIMessage(content='I found two workshops for you.')])
    with patch('app.agent.nodes.memory.save_turn'):
        result = finalize(state)
    assert result['final_response'] == 'I found two workshops for you.'

def test_model_error_is_reported_as_failed_agent_run():
    from contextlib import nullcontext
    session = MagicMock(session_id=uuid.uuid4())
    run = MagicMock(run_id=uuid.uuid4())
    db = MagicMock()
    state = {'final_response': 'I hit a problem reaching the language model.', 'intent': 'GENERAL', 'loop_count': 0, 'tool_calls': [], 'tool_results': [], 'retrieved_documents': [], 'error': 'provider request failed'}
    graph = MagicMock()
    graph.invoke.return_value = state
    with patch('app.agent.graph.load_actor', return_value=MagicMock(role='USER')), patch('app.agent.graph.memory.get_or_create_session', return_value=session), patch('app.agent.graph.start_agent_run', return_value=run), patch('app.agent.graph.get_graph', return_value=graph), patch('app.agent.graph.use_tool_context', return_value=nullcontext()), patch('app.agent.graph.finish_agent_run') as finish_run:
        result = run_agent(db=db, user_id=uuid.uuid4(), message='find an AI workshop')
    assert result.status == 'error'
    assert result.error == 'provider request failed'
    assert finish_run.call_args.kwargs['status'] == 'error'
    assert finish_run.call_args.kwargs['error'] == 'provider request failed'

def test_finalize_produces_a_fallback_when_there_is_no_answer(user_context):
    state = _state(messages=[HumanMessage(content='hi')])
    with patch('app.agent.nodes.memory.save_turn'):
        result = finalize(state)
    assert result['final_response']
    assert 'usable response' in result['final_response'].lower()

def test_finalize_returns_latest_tool_result_when_model_has_no_final_answer(user_context):
    tool_result = ToolMessage(
        content="'AI and Cloud Computing Workshop' has 100 seats and is already booked.",
        tool_call_id='availability-check',
        name='check_venue_availability',
    )
    state = _state(messages=[HumanMessage(content='Create the workshop'), tool_result])

    with patch('app.agent.nodes.memory.save_turn'):
        result = finalize(state)

    assert result['final_response'] == tool_result.content

def test_memory_failure_does_not_lose_the_answer(user_context):
    state = _state(messages=[AIMessage(content='Here you go.')])
    with patch('app.agent.nodes.memory.save_turn', side_effect=RuntimeError('db down')):
        result = finalize(state)
    assert result['final_response'] == 'Here you go.'
    assert user_context.db.rolled_back

def test_system_prompt_states_the_routing_rule():
    prompt = build_system_prompt(role='USER', today='Friday, 18 September 2026')
    assert 'search_event_policy' in prompt
    assert 'USER' in prompt
    assert '18 September 2026' in prompt
    assert 'Answer the latest user message directly' in prompt
    assert 'do not repeat an earlier answer' in prompt
    assert 'map every supplied field to the create_event arguments' in prompt
    assert 'search for that exact venue' in prompt
    assert 'create_event always creates the event as a draft' in prompt

def test_system_prompt_forbids_inventing_ids():
    prompt = build_system_prompt(role='USER', today='today')
    assert 'never invent' in prompt.lower()