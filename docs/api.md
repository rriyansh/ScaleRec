# API Reference

## GET /health
Returns `{"status": "ok"}`.

## GET /recommend/{user_id}

### Response
```json
{
  "user_id": 42,
  "recommendations": [
    {"item_id": 123, "score": 0.87},
    {"item_id": 456, "score": 0.74}
  ],
  "cached": false
}
