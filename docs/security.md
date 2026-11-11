# ScaleRec — Security Guide

## Threat Model

| Threat | Mitigation |
|--------|------------|
| Unauthorized API access | API key auth with constant-time comparison |
| Brute-force key guessing | Rate limiting via SlowAPI |
| Payload flooding | `BodySizeLimitMiddleware` rejects oversize requests |
| Slowloris / timing leaks | Uvicorn timeouts + `X-Process-Time` header |
| Cache poisoning | Cache values validated through Pydantic schemas |
| Dependency failure cascade | Circuit breaker on Redis |
| Info leak in errors | Generic error handler in production |
| Header injection | Security headers middleware |
| PII in logs | Structured logs (request_id only, no payload) |
| Container privilege | Non-root user in Dockerfile |
| Supply chain | Version-pinned deps + `pip install --no-cache-dir` |
| Secret exposure | `Config.to_dict()` redacts secrets |

## Authentication

API key must be provided via:
- `Authorization: Bearer <key>` header, or
- `X-API-Key: <key>` header

Constant-time comparison via `hmac.compare_digest`.

**Generate a production key:**

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set as `API_KEY` env var. Application **refuses to start in production** without it.

## Rate Limiting

Per-IP, per-minute, enforced by SlowAPI:

```yaml
serving:
  rate_limit_per_minute: 120
```

Batch endpoint gets 1/4 the rate (more expensive).

## Request Size Limit

```yaml
serving:
  max_request_bytes: 1048576   # 1 MB
```

Oversized requests return `413` before hitting the handler.

## Security Headers

Every response includes:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: no-referrer`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Cross-Origin-Resource-Policy: same-origin`
- `Cross-Origin-Opener-Policy: same-origin`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`

## CORS

Configure allowed origins explicitly in production:

```yaml
serving:
  cors_origins:
    - "https://app.example.com"
```

Never use `["*"]` in production.

## Error Handling

- `ScaleRecError` subclasses map to specific HTTP status codes
- Unhandled exceptions return generic `500` without stack traces
- All errors include `request_id` for correlation

## Logging

- Structured, one-line per event
- Request ID injected via context var
- **Never** logs API keys, request bodies, or user PII
- Audit middleware logs method + path + status + client IP only

## Model Artifact Integrity

`models_store/version.json` records SHA256 of every artifact. On startup:

- If verification passes → `Model artifacts verified (SHA256)`
- If fails → warning logged (non-fatal so old deployments keep working)

Enable strict mode by changing `Recommender.__init__` to raise on `verify_version() == False`.

## Docker Hardening

- **Multi-stage build** — build tools not in runtime image
- **Non-root user** — UID 1001, no shell
- **Tini as PID 1** — proper signal forwarding
- **Healthcheck** built-in
- **No secrets in image** — everything via env

## Production Checklist

- [ ] `API_KEY` set via secret manager (not `.env` file)
- [ ] `APP_ENV=production` (forces API_KEY requirement)
- [ ] `CONFIG_PATH=config/config.prod.yaml`
- [ ] CORS origins explicitly listed
- [ ] `include_docs: false` (hide Swagger)
- [ ] Redis password set
- [ ] TLS terminated at load balancer
- [ ] Container runs as non-root
- [ ] Health probes configured in orchestrator
- [ ] Metrics scraped by Prometheus
- [ ] Logs shipped to central store
- [ ] Backup strategy for `models_store/`
