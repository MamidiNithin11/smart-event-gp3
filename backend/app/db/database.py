import hashlib
import re
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings
from app.observability.timing import emit_timing, is_chat_run_active

def _get_engine_options(raw_url: str):
    url = raw_url
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    parsed = urlparse(url)
    if parsed.query:
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        if 'channel_binding' in query_params:
            del query_params['channel_binding']
        new_query = urlencode(query_params, doseq=True)
        parsed = parsed._replace(query=new_query)
        url = urlunparse(parsed)
    connect_args = {}
    if url.startswith('postgresql'):
        connect_args = {'keepalives': 1, 'keepalives_idle': 30, 'keepalives_interval': 10, 'keepalives_count': 5}
        if 'sslmode' not in url:
            connect_args['sslmode'] = 'require'
    return (url, connect_args)
db_url, connect_args = _get_engine_options(settings.DATABASE_URL)
engine = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True, pool_recycle=300)

@event.listens_for(engine, 'before_cursor_execute')
def _start_chat_sql_timer(connection, cursor, statement, parameters, context, executemany):
    if not is_chat_run_active():
        return
    timers = connection.info.setdefault('chat_sql_timers', [])
    statement_id = hashlib.sha256(statement.encode('utf-8')).hexdigest()[:12]
    safe_shape = re.sub(r"'(?:''|[^'])*'|\b\d+\b", '?', statement)
    safe_shape = ' '.join(safe_shape.split())[:320]
    timers.append((time.perf_counter(), statement_id, safe_shape))

@event.listens_for(engine, 'after_cursor_execute')
def _finish_chat_sql_timer(connection, cursor, statement, parameters, context, executemany):
    timers = connection.info.get('chat_sql_timers', [])
    if timers:
        started, statement_id, safe_shape = timers.pop()
        emit_timing(
            'db.sql_execute',
            (time.perf_counter() - started) * 1000,
            statement_id=statement_id,
            statement_shape=safe_shape,
            executemany=executemany,
        )

@event.listens_for(engine, 'handle_error')
def _clear_failed_chat_sql_timer(exception_context):
    connection = exception_context.connection
    if connection is not None:
        timers = connection.info.get('chat_sql_timers', [])
        if timers:
            timers.pop()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()