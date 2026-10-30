import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttackProfile:
    weight: float
    severity_p: tuple
    protocols: dict
    dest_ports: dict
    tactic: str
    tags: tuple
    bytes_mult: float
    duration_mult: float
    peak_hours: tuple
    weekend_factor: float
    assets: dict
    exfil_prob: float = 0.0


SEVERITIES = ["Low", "Medium", "High", "Critical"]
SEVERITY_BASE = {"Low": 20, "Medium": 45, "High": 70, "Critical": 90}
BLOCK_PROB = {"Low": 0.88, "Medium": 0.77, "High": 0.57, "Critical": 0.42}
RESPONSE_SCALE = {"Low": 240, "Medium": 120, "High": 45, "Critical": 15}

PROFILES = {
    "DDoS": AttackProfile(
        0.22, (0.10, 0.30, 0.35, 0.25),
        {"UDP": 0.35, "TCP": 0.30, "HTTP": 0.20, "HTTPS": 0.10, "DNS": 0.05},
        {80: 0.35, 443: 0.35, 53: 0.15, 8080: 0.15},
        "Impact", ("flood", "botnet", "amplification", "volumetric", "saturation"),
        6.0, 0.8, tuple(range(0, 24)), 1.1,
        {"Web Server": 0.45, "API Gateway": 0.30, "DNS Server": 0.25},
    ),
    "Phishing": AttackProfile(
        0.20, (0.30, 0.35, 0.25, 0.10),
        {"HTTPS": 0.50, "HTTP": 0.20, "TCP": 0.30},
        {443: 0.60, 80: 0.30, 8080: 0.10},
        "Initial Access", ("spoofed", "credential", "email", "lure", "impersonation"),
        0.2, 0.3, (8, 9, 10, 11, 13, 14, 15, 16), 0.25,
        {"Mail Server": 0.50, "Workstation": 0.50},
        0.05,
    ),
    "Malware": AttackProfile(
        0.16, (0.10, 0.30, 0.35, 0.25),
        {"HTTP": 0.30, "HTTPS": 0.35, "TCP": 0.25, "DNS": 0.10},
        {80: 0.30, 443: 0.40, 8080: 0.20, 53: 0.10},
        "Execution", ("trojan", "payload", "dropper", "infection", "backdoor"),
        1.5, 1.4, tuple(range(7, 20)), 0.5,
        {"Workstation": 0.65, "File Server": 0.20, "Web Server": 0.15},
        0.10,
    ),
    "Brute Force": AttackProfile(
        0.14, (0.30, 0.35, 0.25, 0.10),
        {"TCP": 0.70, "UDP": 0.05, "FTP": 0.25},
        {22: 0.45, 3389: 0.35, 21: 0.20},
        "Credential Access", ("credential", "dictionary", "lockout", "password", "repeated"),
        0.4, 2.0, tuple(range(0, 6)) + tuple(range(22, 24)), 1.3,
        {"Domain Controller": 0.25, "Workstation": 0.35, "Web Server": 0.40},
    ),
    "SQL Injection": AttackProfile(
        0.10, (0.05, 0.20, 0.35, 0.40),
        {"HTTP": 0.45, "HTTPS": 0.55},
        {80: 0.30, 443: 0.45, 8080: 0.15, 3306: 0.10},
        "Exfiltration", ("payload", "database", "exfiltration", "query", "injection"),
        2.5, 0.9, tuple(range(9, 23)), 0.7,
        {"Database": 0.55, "Web Server": 0.30, "API Gateway": 0.15},
        0.35,
    ),
    "Ransomware": AttackProfile(
        0.08, (0.05, 0.20, 0.35, 0.40),
        {"TCP": 0.50, "HTTPS": 0.30, "HTTP": 0.10, "DNS": 0.10},
        {443: 0.40, 3389: 0.30, 80: 0.15, 22: 0.15},
        "Impact", ("encryption", "extortion", "payload", "lockdown", "ransom"),
        4.0, 2.5, tuple(range(0, 6)) + tuple(range(19, 24)), 1.6,
        {"File Server": 0.40, "Domain Controller": 0.20, "Workstation": 0.30, "Database": 0.10},
        0.25,
    ),
    "Port Scan": AttackProfile(
        0.10, (0.45, 0.35, 0.15, 0.05),
        {"TCP": 0.65, "UDP": 0.25, "DNS": 0.10},
        {22: 0.15, 80: 0.15, 443: 0.15, 21: 0.10, 3389: 0.15, 53: 0.10, 8080: 0.10, 3306: 0.10},
        "Reconnaissance", ("reconnaissance", "probe", "enumeration", "scan", "mapping"),
        0.1, 0.4, tuple(range(0, 24)), 1.2,
        {"Web Server": 0.30, "Workstation": 0.20, "Database": 0.15, "Domain Controller": 0.15, "API Gateway": 0.20},
    ),
}

COUNTRIES = {
    "United States": (0.18, (23, 34, 45, 66, 104)),
    "India": (0.14, (14, 27, 49, 117, 122)),
    "China": (0.12, (36, 58, 112, 183, 220)),
    "Russia": (0.10, (31, 46, 77, 95, 188)),
    "Germany": (0.08, (5, 37, 62, 85, 141)),
    "Brazil": (0.07, (131, 177, 186, 189, 200)),
    "United Kingdom": (0.07, (2, 51, 81, 86, 109)),
    "Nigeria": (0.06, (41, 102, 105, 154, 197)),
    "Ukraine": (0.06, (37, 91, 93, 176, 193)),
    "Vietnam": (0.05, (14, 42, 113, 171, 203)),
    "France": (0.04, (15, 78, 82, 90, 151)),
    "Canada": (0.03, (24, 47, 70, 99, 142)),
}

ASSET_SUBNETS = {
    "Web Server": 10, "Database": 11, "Mail Server": 12, "Workstation": 13,
    "Domain Controller": 14, "File Server": 15, "API Gateway": 16, "DNS Server": 17,
}
ASSET_CRITICALITY = {
    "Web Server": 0.7, "Database": 1.0, "Mail Server": 0.6, "Workstation": 0.3,
    "Domain Controller": 1.0, "File Server": 0.8, "API Gateway": 0.7, "DNS Server": 0.6,
}

DETECTION_SOURCES = {"IDS": 0.30, "Firewall": 0.25, "SIEM": 0.20, "EDR": 0.15, "Manual Report": 0.10}
DETECTION_BLOCK_BONUS = {"IDS": 0.03, "Firewall": 0.08, "SIEM": 0.0, "EDR": 0.05, "Manual Report": -0.20}

DESCRIPTION_TEMPLATES = [
    "{attack} incident via {protocol} flagged as {severity} risk against {asset}, tags: {tags}",
    "{severity} severity {attack} activity on {asset} over {protocol} from {country}, tags: {tags}",
    "Detected {attack} ({tactic}) targeting {asset} using {protocol}; indicators: {tags}",
    "{country}-origin {attack} traffic hit {asset} via {protocol}, rated {severity}, tags: {tags}",
]

NULLABLE_COLUMNS = ["country", "protocol", "duration_sec", "detection_source", "response_time_min"]


@dataclass
class Config:
    rows: int = 2000
    seed: int = 42
    start_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1))
    days: int = 365
    output: Path = Path("data/security_incidents.csv")
    fmt: str = "csv"
    missing_rate: float = 0.0
    campaigns: int = 6
    anomaly_rate: float = 0.01


def parse_args():
    p = argparse.ArgumentParser(description="Synthetic security incident generator")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-date", type=str, default="2025-01-01")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--output", type=str, default="data/security_incidents.csv")
    p.add_argument("--format", dest="fmt", choices=["csv", "json", "parquet"], default=None)
    p.add_argument("--missing-rate", type=float, default=0.0)
    p.add_argument("--campaigns", type=int, default=6)
    p.add_argument("--anomaly-rate", type=float, default=0.01)
    a = p.parse_args()
    out = Path(a.output)
    fmt = a.fmt or (out.suffix.lstrip(".") if out.suffix.lstrip(".") in {"csv", "json", "parquet"} else "csv")
    return Config(
        rows=a.rows,
        seed=a.seed,
        start_date=datetime.strptime(a.start_date, "%Y-%m-%d"),
        days=a.days,
        output=out,
        fmt=fmt,
        missing_rate=a.missing_rate,
        campaigns=a.campaigns,
        anomaly_rate=a.anomaly_rate,
    )


def normalized(weights):
    arr = np.array(list(weights), dtype=float)
    return arr / arr.sum()


def weighted_choice(rng, mapping):
    keys = list(mapping.keys())
    return keys[rng.choice(len(keys), p=normalized(mapping.values()))]


def build_day_weights(rng, cfg):
    weights = np.ones(cfg.days)
    trend = 1 + 0.4 * np.linspace(0, 1, cfg.days)
    weights = weights * trend
    campaigns = []
    for _ in range(cfg.campaigns):
        centre = int(rng.integers(5, max(6, cfg.days - 5)))
        width = int(rng.integers(2, 6))
        attack = rng.choice(list(PROFILES.keys()))
        lift = rng.uniform(3, 8)
        for d in range(max(0, centre - width), min(cfg.days, centre + width + 1)):
            weights[d] += lift * np.exp(-((d - centre) ** 2) / (2 * (width / 2 + 0.5) ** 2))
        campaigns.append({"centre": centre, "width": width, "attack": attack, "lift": lift})
    return weights, campaigns


def random_public_ip(rng, country):
    first = int(rng.choice(COUNTRIES[country][1]))
    return f"{first}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def random_internal_ip(rng, asset):
    return f"10.{ASSET_SUBNETS[asset]}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def pick_timestamp(rng, cfg, day_weights, campaign_days, attack, profile):
    day_p = day_weights.copy()
    if attack in campaign_days:
        for d, lift in campaign_days[attack].items():
            day_p[d] += lift * 3
    for _ in range(20):
        day = int(rng.choice(cfg.days, p=day_p / day_p.sum()))
        date = cfg.start_date + timedelta(days=day)
        keep = 1.0 if date.weekday() < 5 else profile.weekend_factor
        if rng.random() <= min(keep, 1.0):
            break
    hour_p = np.full(24, 0.35)
    for h in profile.peak_hours:
        hour_p[h] = 1.0
    hour = int(rng.choice(24, p=hour_p / hour_p.sum()))
    return date + timedelta(hours=hour, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))


def generate_row(i, rng, cfg, day_weights, campaign_days, attack_names, attack_probs, country_names, country_probs):
    attack = attack_names[rng.choice(len(attack_names), p=attack_probs)]
    profile = PROFILES[attack]

    severity = SEVERITIES[rng.choice(4, p=normalized(profile.severity_p))]
    sev_base = SEVERITY_BASE[severity]

    country = country_names[rng.choice(len(country_names), p=country_probs)]
    protocol = weighted_choice(rng, profile.protocols)
    asset = weighted_choice(rng, profile.assets)
    dest_port = int(weighted_choice(rng, profile.dest_ports))
    source = weighted_choice(rng, DETECTION_SOURCES)

    timestamp = pick_timestamp(rng, cfg, day_weights, campaign_days, attack, profile)

    duration = float(np.clip(rng.exponential(sev_base * 2 * profile.duration_mult), 1, 3600))
    bytes_tx = float(np.clip(rng.lognormal(np.log(sev_base * 4000 * profile.bytes_mult), 0.9), 100, 5_000_000))
    avg_packet = rng.uniform(200, 800) if attack != "DDoS" else rng.uniform(60, 300)
    packets = int(np.clip(bytes_tx / avg_packet, 1, 20000))

    exfil = bool(rng.random() < profile.exfil_prob * (sev_base / 60))
    criticality = ASSET_CRITICALITY[asset]
    repeat_offender = bool(rng.random() < 0.12)
    failed_logins = int(rng.poisson(40 if attack == "Brute Force" else 1))
    anomaly = bool(rng.random() < cfg.anomaly_rate)

    if anomaly:
        bytes_tx = float(np.clip(bytes_tx * rng.uniform(5, 15), 100, 5_000_000))
        duration = float(np.clip(duration * rng.uniform(3, 8), 1, 3600))
        packets = int(np.clip(packets * rng.uniform(3, 10), 1, 20000))

    risk = rng.normal(sev_base, 7)
    risk += 6 * criticality + (5 if exfil else 0) + (4 if repeat_offender else 0) + (4 if anomaly else 0)
    risk += min(failed_logins, 100) * 0.05
    risk = float(np.clip(risk, 1, 100))

    false_positive = bool(rng.random() < (0.07 if severity == "Low" else 0.03 if severity == "Medium" else 0.01))
    if false_positive:
        risk = float(np.clip(risk * rng.uniform(0.4, 0.7), 1, 100))

    block_p = np.clip(BLOCK_PROB[severity] + DETECTION_BLOCK_BONUS[source] - 0.05 * criticality, 0.05, 0.98)
    blocked = bool(rng.random() < block_p)

    response_min = float(np.clip(rng.exponential(RESPONSE_SCALE[severity]) * (0.6 if blocked else 1.4), 1, 4320))

    if false_positive:
        status = "False Positive"
    else:
        status = weighted_choice(rng, {"Resolved": 0.62, "Investigating": 0.2, "Open": 0.1, "Escalated": 0.08})
        if severity == "Critical" and status == "Resolved" and rng.random() < 0.25:
            status = "Escalated"

    resolved_at = None
    if status in ("Resolved", "False Positive"):
        resolved_at = timestamp + timedelta(minutes=response_min + float(rng.exponential(90)))

    tags = rng.choice(profile.tags, size=3, replace=False)
    template = DESCRIPTION_TEMPLATES[rng.integers(0, len(DESCRIPTION_TEMPLATES))]
    description = template.format(
        attack=attack, protocol=protocol, severity=severity.lower(), asset=asset.lower(),
        country=country, tactic=profile.tactic, tags=" ".join(tags),
    )

    return {
        "incident_id": f"INC-{i + 1:05d}",
        "date": timestamp,
        "attack_type": attack,
        "mitre_tactic": profile.tactic,
        "severity": severity,
        "country": country,
        "source_ip": random_public_ip(rng, country),
        "dest_ip": random_internal_ip(rng, asset),
        "target_asset": asset,
        "asset_criticality": criticality,
        "protocol": protocol,
        "source_port": int(rng.integers(1024, 65535)),
        "dest_port": dest_port,
        "duration_sec": round(duration, 2),
        "bytes_transferred": round(bytes_tx, 2),
        "packets": packets,
        "failed_logins": failed_logins,
        "data_exfiltrated": exfil,
        "repeat_offender": repeat_offender,
        "detection_source": source,
        "risk_score": round(risk, 2),
        "blocked": blocked,
        "response_time_min": round(response_min, 1),
        "status": status,
        "resolved_at": resolved_at,
        "is_false_positive": false_positive,
        "is_anomaly": anomaly,
        "description": description,
    }


def inject_missing(df, rng, rate):
    if rate <= 0:
        return df
    for col in NULLABLE_COLUMNS:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def validate(df, cfg):
    window_end = cfg.start_date + timedelta(days=cfg.days + 1)
    checks = {
        "unique incident ids": df["incident_id"].is_unique,
        "risk score in [1, 100]": df["risk_score"].between(1, 100).all(),
        "ports in valid range": df["source_port"].between(1, 65535).all() and df["dest_port"].between(1, 65535).all(),
        "positive packets": (df["packets"] >= 1).all(),
        "resolved after detected": (df["resolved_at"].dropna() >= df.loc[df["resolved_at"].notna(), "date"]).all(),
        "dates within window": bool(((df["date"] >= cfg.start_date) & (df["date"] < window_end)).all()),
    }
    print("\nValidation:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


def report(df, campaigns):
    print("\nIncidents by attack type:")
    print(df["attack_type"].value_counts().to_string())

    print("\nSeverity mix by attack type (row %):")
    print((pd.crosstab(df["attack_type"], df["severity"], normalize="index")[SEVERITIES] * 100).round(1).to_string())

    print("\nBlock rate and mean risk by severity:")
    print(df.groupby("severity")[["blocked", "risk_score", "response_time_min"]].mean().reindex(SEVERITIES).round(3).to_string())

    print("\nTop 5 source countries:")
    print(df["country"].value_counts().head(5).to_string())

    print("\nTop 5 target assets:")
    print(df["target_asset"].value_counts().head(5).to_string())

    print("\nStatus breakdown:")
    print(df["status"].value_counts().to_string())

    monthly = df.set_index("date").resample("MS").size()
    print("\nIncidents per month:")
    print(monthly.to_string())

    print(f"\nData exfiltration incidents: {int(df['data_exfiltrated'].sum())}")
    print(f"Anomalous incidents: {int(df['is_anomaly'].sum())}")
    print(f"Planted campaigns: {len(campaigns)}")
    for c in campaigns:
        print(f"  day {c['centre']:>3} +/- {c['width']}  {c['attack']:<14} lift={c['lift']:.1f}")


def write(df, cfg):
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["resolved_at"] = pd.to_datetime(out["resolved_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if cfg.fmt == "csv":
        out.to_csv(cfg.output, index=False)
    elif cfg.fmt == "json":
        out.to_json(cfg.output, orient="records", indent=2)
    else:
        out.to_parquet(cfg.output, index=False)


def main():
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    attack_names = list(PROFILES.keys())
    attack_probs = normalized(p.weight for p in PROFILES.values())
    country_names = list(COUNTRIES.keys())
    country_probs = normalized(v[0] for v in COUNTRIES.values())

    day_weights, campaigns = build_day_weights(rng, cfg)
    campaign_days = {}
    for c in campaigns:
        bucket = campaign_days.setdefault(c["attack"], {})
        for d in range(max(0, c["centre"] - c["width"]), min(cfg.days, c["centre"] + c["width"] + 1)):
            bucket[d] = bucket.get(d, 0) + c["lift"]

    rows = [
        generate_row(i, rng, cfg, day_weights, campaign_days, attack_names, attack_probs, country_names, country_probs)
        for i in range(cfg.rows)
    ]

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["incident_id"] = [f"INC-{i + 1:05d}" for i in range(len(df))]
    df["date"] = pd.to_datetime(df["date"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])

    ok = validate(df, cfg)
    report(df, campaigns)

    df = inject_missing(df, rng, cfg.missing_rate)
    write(df, cfg)

    meta = {
        "rows": len(df),
        "seed": cfg.seed,
        "start_date": cfg.start_date.strftime("%Y-%m-%d"),
        "days": cfg.days,
        "missing_rate": cfg.missing_rate,
        "anomaly_rate": cfg.anomaly_rate,
        "campaigns": campaigns,
        "validation_passed": ok,
    }
    meta_path = cfg.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    print(f"\nGenerated {len(df)} rows -> {cfg.output}")
    print(f"Metadata -> {meta_path}")
    print(df.head())


if __name__ == "__main__":
    main()
