# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, send an email to **gushuchun123@gmail.com**. We will respond as soon
as possible and work with you on a fix and coordinated disclosure timeline.

## Security Considerations for Raglan Users

Raglan is a retrieval engine — it searches through your documents.
Keep these security practices in mind when deploying:

1. **PII in documents**: Use the ``LoggingMiddleware`` with log-level filtering
   to avoid writing sensitive content to application logs.  Search traces may
   contain query text; consider the ``TraceLevel`` enum when exposing trace data.

2. **API keys**: Pass API keys via environment variables or secret management
   services, never hardcode them in configuration files.

3. **Vector database access**: When using read-only retrieval (e.g.
   ``ConfigurablePgvectorRetriever``), use minimal database credentials.
   Retrievers that support indexing (``ChromaDBRetriever``, ``QdrantRetriever``)
   may write to your database — ensure appropriate access controls.

4. **Trace data**: The `debug` and `full` trace levels may contain raw search
   queries and document content. Use `minimal` level in production.
