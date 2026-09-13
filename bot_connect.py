import ccxt

# Verbindung mit deinen persönlichen API-Schlüsseln herstellen
exchange = ccxt.binance({
    "apiKey": "adRZ1PONLj0OV05VMVdqZ9PAoG0tuqGWHXzYyrLDiVL1lY9ZeDXi9PLlW4pDpPtN",
    "secret": "LWJ0WyLC0M2f3M0wARfm83vOoSitNJm8rkLylHYWfgknDhZPLIW2EaFZVtYsI43N",
    "options": {
        "defaultType": "spot"
    },  # Legt fest, dass im Spot-Markt gehandelt wird
})

# Testen, ob die Verbindung klappt (z.B. Kontostand abfragen)
try:
  balance = exchange.fetch_balance()
  print("Verbindung erfolgreich! USDT-Guthaben:", balance["free"].get("USDT", 0))
except Exception as e:
  print("Verbindungsfehler:", e)