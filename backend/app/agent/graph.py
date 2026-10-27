from __future__ import annotations
import logging
import time
from uuid import UUID, uuid4
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session
from app.agent import memory
from app.agent.nodes import act, after_act, after_load_memory, create_structured_event, fast_registration_count, fast_venue_directory, finalize, load_memory, reason, should_continue
from app.agent.state import AgentState, new_state
from app.core.permissions import GUEST_ROLE, load_actor
from app.observability.logger import finish_agent_run, start_agent_run
from app.observability.timing import bind_chat_run, current_llm_call_count, current_llm_invocation_count, emit_timing
from app.tools.context import ToolContext, use_tool_context

logger = logging.getLogger(__name__)
_compiled_graph = None

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('load_memory', load_memory)
    graph.add_node('registration_count', fast_registration_count)
    graph.add_node('venue_directory', fast_venue_directory)
    graph.add_node('reason', reason)
    graph.add_node('act', act)
    graph.add_node('structured_event_create', create_structured_event)
    graph.add_node('finalize', finalize)
    graph.add_edge(START, 'load_memory')
    graph.add_conditional_edges(
        'load_memory',
        after_load_memory,
        {
            'registration_count': 'registration_count',
            'venue_directory': 'venue_directory',
            'reason': 'reason',
        },
    )
    graph.add_edge('registration_count', 'finalize')
    graph.add_edge('venue_directory', 'finalize')
    graph.add_conditional_edges('reason', should_continue, {'act': 'act', 'finalize': 'finalize'})
    graph.add_conditional_edges(
        'act',
        after_act,
        {
            'reason': 'reason',
            'finalize': 'finalize',
            'structured_event_create': 'structured_event_create',
        },
    )
    graph.add_edge('structured_event_create', 'finalize')
    graph.add_edge('finalize', END)
    return graph.compile()

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info('Agent graph compiled')
    return _compiled_graph

class AgentResult:

    def __init__(self, *, reply: str, session_id: UUID, run_id: UUID, intent: str, tools_used: list[str], tool_results: list[dict], sources: list[str], latency_ms: float, status: str, error: str=''):
        self.reply = reply
        self.session_id = session_id
        self.run_id = run_id
        self.intent = intent
        self.tools_used = tools_used
        self.tool_results = tool_results
        self.sources = sources
        self.latency_ms = latency_ms
        self.status = status
        self.error = error

def run_agent(db: Session, user_id: UUID | None, message: str, session_id: UUID | None=None) -> AgentResult:
    started = time.perf_counter()
    stage_started = time.perf_counter()
    actor = load_actor(db, user_id) if user_id is not None else None
    emit_timing('db.actor_lookup', (time.perf_counter() - stage_started) * 1000)
    role = actor.role if actor is not None else GUEST_ROLE
    stage_started = time.perf_counter()
    session = memory.get_or_create_session(db=db, user_id=user_id, session_id=session_id, first_message=message)
    emit_timing(
        'db.session_lookup_or_create',
        (time.perf_counter() - stage_started) * 1000,
        existing_session=bool(session_id),
    )
    stage_started = time.perf_counter()
    run = start_agent_run(db=db, session_id=session.session_id, user_id=user_id, user_request=message)
    emit_timing('db.agent_run_create', (time.perf_counter() - stage_started) * 1000)
    stage_started = time.perf_counter()
    db.commit()
    emit_timing('db.start_commit', (time.perf_counter() - stage_started) * 1000)
    state = new_state(user_id=user_id, session_id=session.session_id, run_id=run.run_id, message=message)
    context = ToolContext(db=db, user_id=user_id, role=role)
    try:
        with bind_chat_run(run.run_id), use_tool_context(context):
            graph_started = time.perf_counter()
            final_state = get_graph().invoke(state)
            emit_timing('langgraph.total', (time.perf_counter() - graph_started) * 1000)
            emit_timing(
                'llm.calls_per_request',
                0,
                call_count=current_llm_call_count(),
                router_invocations=current_llm_invocation_count(),
                tool_loops=final_state.get('loop_count', 0),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        reply = final_state.get('final_response', '')
        tools_used = [call['name'] for call in final_state.get('tool_calls', [])]
        error = final_state.get('error', '')
        status = 'error' if error else 'success'
        persistence_started = time.perf_counter()
        finish_agent_run(db=db, run=run, final_response=reply, intent=final_state.get('intent', ''), tool_loop_count=final_state.get('loop_count', 0), latency_ms=latency_ms, status=status, error=error or None)
        db.commit()
        emit_timing('db.finish_run_and_commit', (time.perf_counter() - persistence_started) * 1000, run_id=str(run.run_id))
        emit_timing('agent.total', (time.perf_counter() - started) * 1000, run_id=str(run.run_id), status=status)
        logger.info('run=%s status=%s intent=%s tools=%s latency=%.0fms', run.run_id, status, final_state.get('intent'), tools_used, latency_ms)
        return AgentResult(reply=reply, session_id=session.session_id, run_id=run.run_id, intent=final_state.get('intent', ''), tools_used=tools_used, tool_results=final_state.get('tool_results', []), sources=final_state.get('retrieved_documents', []), latency_ms=latency_ms, status=status, error=error)
    except Exception as exc:
        logger.exception('Agent run %s failed', run.run_id)
        latency_ms = (time.perf_counter() - started) * 1000
        emit_timing('agent.total', latency_ms, run_id=str(run.run_id), status='error')
        db.rollback()
        try:
            finish_agent_run(db=db, run=run, final_response='', intent='', tool_loop_count=0, latency_ms=latency_ms, status='error', error=str(exc))
            db.commit()
        except Exception:
            db.rollback()
            logger.exception('Could not record the failed run')
        return AgentResult(reply='Something went wrong while handling that request. Please try again, or contact an administrator if it keeps happening.', session_id=session.session_id, run_id=run.run_id, intent='', tools_used=[], tool_results=[], sources=[], latency_ms=latency_ms, status='error', error=str(exc))