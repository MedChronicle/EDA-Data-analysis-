import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 2000

attack_types = ["DDoS", "Phishing", "Malware", "Brute Force", "SQL Injection", "Ransomware", "Port Scan"]
attack_weights = [0.22, 0.20, 0.16, 0.14, 0.10, 0.08, 0.10]

severity_levels = ["Low", "Medium", "High", "Critical"]

countries = ["United States", "India", "China", "Russia", "Germany", "Brazil",
             "United Kingdom", "Nigeria", "Ukraine", "Vietnam", "France", "Canada"]
country_weights = [0.18, 0.14, 0.12, 0.10, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.04, 0.03]

protocols = ["TCP", "UDP", "HTTP", "HTTPS", "DNS", "FTP"]

tag_pool = {
    "DDoS": ["flood", "botnet", "amplification", "volumetric", "saturation"],
    "Phishing": ["spoofed", "credential", "email", "lure", "impersonation"],
    "Malware": ["trojan", "payload", "dropper", "infection", "backdoor"],
    "Brute Force": ["credential", "dictionary", "lockout", "password", "repeated"],
    "SQL Injection": ["payload", "database", "exfiltration", "query", "injection"],
    "Ransomware": ["encryption", "extortion", "payload", "lockdown", "ransom"],
    "Port Scan": ["reconnaissance", "probe", "enumeration", "scan", "mapping"],
}

start_date = datetime(2025, 1, 1)

rows = []
for i in range(N):
    attack = np.random.choice(attack_types, p=attack_weights)

    if attack in ["Ransomware", "SQL Injection"]:
        severity_p = [0.05, 0.20, 0.35, 0.40]
    elif attack in ["DDoS", "Malware"]:
        severity_p = [0.10, 0.30, 0.35, 0.25]
    else:
        severity_p = [0.30, 0.35, 0.25, 0.10]
    severity = np.random.choice(severity_levels, p=severity_p)

    severity_base = {"Low": 20, "Medium": 45, "High": 70, "Critical": 90}[severity]
    risk_score = np.clip(np.random.normal(severity_base, 8), 1, 100)

    country = np.random.choice(countries, p=country_weights)
    protocol = np.random.choice(protocols)

    duration_sec = np.clip(np.random.exponential(scale=severity_base * 2), 1, 3600)
    bytes_transferred = np.clip(np.random.exponential(scale=severity_base * 5000), 100, 5_000_000)
    packets = int(np.clip(bytes_transferred / np.random.uniform(200, 800), 1, 20000))

    source_port = np.random.randint(1024, 65535)
    dest_port = np.random.choice([80, 443, 22, 21, 3389, 53, 8080, 3306])

    blocked_prob = {"Low": 0.85, "Medium": 0.75, "High": 0.55, "Critical": 0.40}[severity]
    blocked = np.random.rand() < blocked_prob

    day_offset = np.random.randint(0, 365)
    hour = np.random.randint(0, 24)
    date = start_date + timedelta(days=day_offset, hours=int(hour))

    tags = np.random.choice(tag_pool[attack], size=3, replace=False)
    description = f"{attack} incident via {protocol} flagged as {severity.lower()} risk, tags: {' '.join(tags)}"

    rows.append({
        "incident_id": f"INC-{i+1:05d}",
        "date": date.strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type": attack,
        "severity": severity,
        "country": country,
        "protocol": protocol,
        "source_port": source_port,
        "dest_port": dest_port,
        "duration_sec": round(duration_sec, 2),
        "bytes_transferred": round(bytes_transferred, 2),
        "packets": packets,
        "risk_score": round(risk_score, 2),
        "blocked": blocked,
        "description": description,
    })

df = pd.DataFrame(rows)
df.to_csv("data/security_incidents.csv", index=False)
print(f"Generated {len(df)} rows -> data/security_incidents.csv")
print(df.head())
