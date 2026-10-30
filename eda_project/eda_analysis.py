import os
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000

rng = np.random.default_rng(SEED)

PROFILES = {
    "Phishing":          {"dur": 40,    "bytes": 2e4,  "pkts": 30,    "risk": 45, "w": 0.20},
    "Malware":           {"dur": 600,   "bytes": 5e6,  "pkts": 4000,  "risk": 70, "w": 0.16},
    "Ransomware":        {"dur": 1800,  "bytes": 5e7,  "pkts": 20000, "risk": 90, "w": 0.07},
    "DDoS":              {"dur": 900,   "bytes": 8e8,  "pkts": 9e5,   "risk": 75, "w": 0.12},
    "SQL Injection":     {"dur": 120,   "bytes": 3e5,  "pkts": 400,   "risk": 65, "w": 0.11},
    "Brute Force":       {"dur": 1500,  "bytes": 8e5,  "pkts": 15000, "risk": 50, "w": 0.13},
    "Man-in-the-Middle": {"dur": 700,   "bytes": 4e6,  "pkts": 6000,  "risk": 60, "w": 0.06},
    "Insider Threat":    {"dur": 3600,  "bytes": 2e8,  "pkts": 12000, "risk": 80, "w": 0.05},
    "Zero-Day Exploit":  {"dur": 300,   "bytes": 1e7,  "pkts": 2500,  "risk": 95, "w": 0.04},
    "Port Scan":         {"dur": 60,    "bytes": 5e4,  "pkts": 2000,  "risk": 30, "w": 0.06},
}

COUNTRIES = [
    "United States", "India", "China", "Russia", "Brazil", "Germany",
    "United Kingdom", "France", "Japan", "Canada", "Australia", "South Korea",
    "Nigeria", "South Africa", "Mexico", "Italy", "Spain", "Netherlands",
    "Singapore", "United Arab Emirates", "Turkey", "Indonesia", "Argentina", "Poland",
]
COUNTRY_WEIGHTS = np.array([
    18, 12, 10, 7, 5, 6, 6, 4, 4, 3, 3, 3,
    2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1,
], dtype=float)
COUNTRY_WEIGHTS /= COUNTRY_WEIGHTS.sum()

DESCRIPTIONS = {
    "Phishing": [
        "deceptive email campaign targeting employee credentials",
        "spoofed login page harvesting user passwords",
        "spear phishing message impersonating executive request",
        "fraudulent invoice link delivering credential stealer",
    ],
    "Malware": [
        "trojan payload detected on endpoint after suspicious download",
        "worm propagating across internal network shares",
        "spyware beaconing to external command server",
        "infected attachment executed malicious macro on workstation",
    ],
    "Ransomware": [
        "file encryption activity detected with ransom note dropped",
        "ransomware locked shared drives and demanded cryptocurrency payment",
        "mass file renaming and encryption across backup servers",
    ],
    "DDoS": [
        "volumetric flood overwhelmed web server bandwidth",
        "botnet traffic surge caused service outage on public portal",
        "SYN flood exhausted connection table on edge firewall",
        "distributed traffic spike degraded application availability",
    ],
    "SQL Injection": [
        "malicious query payload attempted database table extraction",
        "unsanitized input exploited on login form to bypass authentication",
        "injection attempt against customer database endpoint blocked",
    ],
    "Brute Force": [
        "repeated failed login attempts against admin account",
        "password guessing attack on remote desktop service",
        "credential stuffing using leaked username password combinations",
        "automated SSH login attempts from single source address",
    ],
    "Man-in-the-Middle": [
        "intercepted session traffic via rogue access point",
        "forged certificate used to eavesdrop on encrypted connection",
        "ARP spoofing redirected internal traffic through attacker host",
    ],
    "Insider Threat": [
        "privileged user exfiltrated sensitive records to personal storage",
        "unauthorized access to confidential files outside working hours",
        "employee copied customer data onto removable device",
    ],
    "Zero-Day Exploit": [
        "unknown vulnerability exploited for remote code execution",
        "unpatched application flaw abused to gain system privileges",
        "novel exploit chain bypassed endpoint protection controls",
    ],
    "Port Scan": [
        "sequential probing of open ports across server subnet",
        "reconnaissance scan enumerating exposed services",
        "stealth scan mapping firewall rules and active hosts",
    ],
}

names = list(PROFILES.keys())
weights = np.array([PROFILES[n]["w"] for n in names])
weights /= weights.sum()

attack_type = rng.choice(names, size=N_ROWS, p=weights)
country = rng.choice(COUNTRIES, size=N_ROWS, p=COUNTRY_WEIGHTS)

start = pd.Timestamp("2024-01-01")
end = pd.Timestamp("2025-12-31")
span_days = (end - start).days
date = start + pd.to_timedelta(rng.integers(0, span_days + 1, size=N_ROWS), unit="D")

duration = np.empty(N_ROWS)
bytes_transferred = np.empty(N_ROWS)
packets = np.empty(N_ROWS)
base_risk = np.empty(N_ROWS)
description = []

for i, atk in enumerate(attack_type):
    p = PROFILES[atk]
    duration[i] = rng.lognormal(np.log(p["dur"]), 0.6)
    bytes_transferred[i] = rng.lognormal(np.log(p["bytes"]), 0.8)
    packets[i] = rng.lognormal(np.log(p["pkts"]), 0.7)
    base_risk[i] = p["risk"]
    description.append(rng.choice(DESCRIPTIONS[atk]))

size_effect = 4 * (np.log10(bytes_transferred) - np.log10(bytes_transferred).mean())
duration_effect = 3 * (np.log10(duration) - np.log10(duration).mean())
risk_score = base_risk + size_effect + duration_effect + rng.normal(0, 6, N_ROWS)
risk_score = np.clip(risk_score, 1, 100).round(2)

severity = pd.cut(
    risk_score,
    bins=[0, 40, 62, 80, 100],
    labels=["Low", "Medium", "High", "Critical"],
    include_lowest=True,
).astype(str)

df = pd.DataFrame({
    "incident_id": [f"INC-{100000 + i}" for i in range(N_ROWS)],
    "date": date,
    "country": country,
    "attack_type": attack_type,
    "severity": severity,
    "duration_sec": duration.round(2),
    "bytes_transferred": bytes_transferred.round(0).astype(np.int64),
    "packets": packets.round(0).astype(np.int64),
    "risk_score": risk_score,
    "description": description,
}).sort_values("date").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
df.to_csv("data/security_incidents.csv", index=False)

print(f"Saved {len(df)} rows to data/security_incidents.csv")
print(df["severity"].value_counts())
print(df.head())
