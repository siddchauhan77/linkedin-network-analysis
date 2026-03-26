#!/usr/bin/env python3
"""LinkedIn Network Analysis — Sidd Chauhan"""

import pandas as pd
import csv
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent
TODAY = datetime(2026, 3, 25)
NINETY_DAYS_AGO = TODAY - timedelta(days=90)

# --- Seniority scoring ---
SENIORITY = {
    'high': ['ceo', 'cfo', 'cto', 'coo', 'cmo', 'cpo', 'cro', 'chief',
             'vp', 'vice president', 'svp', 'evp', 'partner', 'founder',
             'co-founder', 'cofounder', 'president', 'owner', 'principal',
             'managing director', 'general manager', 'head of', 'general partner'],
    'medium': ['director', 'senior director', 'sr director', 'manager',
               'senior manager', 'sr manager', 'lead', 'team lead'],
}

def seniority_score(title):
    if not title or not isinstance(title, str):
        return 0, 'unknown'
    t = title.lower().strip()
    for word in SENIORITY['high']:
        if word in t:
            return 3, 'high'
    for word in SENIORITY['medium']:
        if word in t:
            return 2, 'medium'
    return 1, 'standard'


def parse_date_flexible(d):
    """Parse dates in various LinkedIn formats."""
    if not d or not isinstance(d, str):
        return None
    d = d.strip()
    for fmt in ['%d %b %Y', '%m/%d/%y, %I:%M %p', '%Y-%m-%d %H:%M:%S %Z',
                '%Y-%m-%d', '%b %Y', '%Y']:
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


# =========================================================================
# LOAD DATA
# =========================================================================
def load_connections():
    """Load Connections.csv, skipping the notes header."""
    lines = open(BASE / 'Connections.csv', 'r', encoding='utf-8-sig').readlines()
    # Find the actual header line (starts with "First Name")
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('First Name'):
            start = i
            break
    from io import StringIO
    return pd.read_csv(StringIO(''.join(lines[start:])))


def load_messages():
    return pd.read_csv(BASE / 'messages.csv', encoding='utf-8-sig')


def load_invitations():
    return pd.read_csv(BASE / 'Invitations.csv', encoding='utf-8-sig')


def load_job_applications():
    frames = []
    for f in sorted(BASE.glob('Jobs/Job Applications*.csv')):
        frames.append(pd.read_csv(f, encoding='utf-8-sig'))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_saved_jobs():
    frames = []
    for f in sorted(BASE.glob('Jobs/Saved Jobs*.csv')):
        frames.append(pd.read_csv(f, encoding='utf-8-sig'))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================================================================
# SECTION 1: NETWORK OVERVIEW
# =========================================================================
def network_overview(conn):
    lines = []
    lines.append('## 1. Network Overview\n')
    lines.append(f'**Total connections: {len(conn)}**\n')

    # Parse dates
    conn['date'] = conn['Connected On'].apply(parse_date_flexible)
    conn['year_month'] = conn['date'].apply(lambda d: d.strftime('%Y-%m') if d else None)

    # Growth by month (last 12 months)
    recent = conn[conn['date'] >= datetime(2025, 3, 1)].copy()
    monthly = recent.groupby('year_month').size().sort_index()
    if len(monthly):
        lines.append('### Connection Growth (Last 12 Months)\n')
        lines.append('| Month | New Connections |')
        lines.append('|-------|----------------|')
        for m, c in monthly.items():
            lines.append(f'| {m} | {c} |')
        lines.append('')

    # Top companies
    companies = conn['Company'].dropna().str.strip()
    top_companies = companies.value_counts().head(25)
    lines.append('### Top 25 Companies in Your Network\n')
    lines.append('| # | Company | Connections |')
    lines.append('|---|---------|-------------|')
    for i, (company, count) in enumerate(top_companies.items(), 1):
        lines.append(f'| {i} | {company} | {count} |')
    lines.append('')

    # Top titles
    titles = conn['Position'].dropna().str.strip()
    top_titles = titles.value_counts().head(25)
    lines.append('### Top 25 Titles\n')
    lines.append('| # | Title | Count |')
    lines.append('|---|-------|-------|')
    for i, (title, count) in enumerate(top_titles.items(), 1):
        lines.append(f'| {i} | {title} | {count} |')
    lines.append('')

    # Industry clustering by title keywords
    clusters = defaultdict(int)
    keyword_map = {
        'Product': ['product manager', 'product', 'apm', 'product analyst', 'product owner', 'product lead', 'product strategy'],
        'Engineering': ['engineer', 'developer', 'software', 'swe', 'backend', 'frontend', 'fullstack', 'full stack', 'devops', 'sre'],
        'Data & Analytics': ['data', 'analytics', 'analyst', 'data scientist', 'machine learning', 'ml', 'ai ', 'artificial intelligence', 'business intelligence'],
        'Design & UX': ['design', 'ux', 'ui', 'user experience', 'user interface', 'graphic design', 'creative'],
        'Marketing': ['marketing', 'growth', 'seo', 'content', 'brand', 'copywriter', 'social media'],
        'Sales & BD': ['sales', 'business development', 'account executive', 'account manager', 'bdr', 'sdr'],
        'Leadership': ['ceo', 'cfo', 'cto', 'coo', 'vp', 'vice president', 'chief', 'founder', 'president', 'partner'],
        'Recruiting & HR': ['recruiter', 'recruiting', 'talent', 'human resources', 'hr ', 'people operations'],
        'Consulting': ['consultant', 'consulting', 'advisory'],
        'Finance': ['finance', 'financial', 'investment', 'banking', 'accountant', 'accounting'],
    }
    for title in titles:
        t = title.lower()
        matched = False
        for cluster, keywords in keyword_map.items():
            if any(k in t for k in keywords):
                clusters[cluster] += 1
                matched = True
                break
        if not matched:
            clusters['Other'] += 1

    lines.append('### Network by Function/Industry\n')
    lines.append('| Function | Connections |')
    lines.append('|----------|-------------|')
    for cluster, count in sorted(clusters.items(), key=lambda x: -x[1]):
        lines.append(f'| {cluster} | {count} |')
    lines.append('')

    return '\n'.join(lines)


# =========================================================================
# SECTION 2: INFLUENCE & HIGH-VALUE CONTACTS
# =========================================================================
def influence_scoring(conn):
    lines = []
    lines.append('## 2. Influence & High-Value Contacts\n')

    scored = []
    for _, row in conn.iterrows():
        score, level = seniority_score(row.get('Position'))
        scored.append({
            'Name': f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
            'Company': row.get('Company', ''),
            'Position': row.get('Position', ''),
            'Score': score,
            'Level': level,
            'Email': row.get('Email Address', ''),
            'URL': row.get('URL', ''),
        })

    scored.sort(key=lambda x: -x['Score'])

    # Counts by level
    level_counts = Counter(s['Level'] for s in scored)
    lines.append(f"- **High seniority** (C-suite, VP, Founder, Partner): **{level_counts.get('high', 0)}**")
    lines.append(f"- **Mid seniority** (Director, Manager, Lead): **{level_counts.get('medium', 0)}**")
    lines.append(f"- **Standard**: **{level_counts.get('standard', 0)}**")
    lines.append(f"- **Unknown/blank title**: **{level_counts.get('unknown', 0)}**")
    lines.append('')

    # Top 50
    lines.append('### Top 50 Highest-Leverage Contacts\n')
    lines.append('| # | Name | Company | Position | Level |')
    lines.append('|---|------|---------|----------|-------|')
    for i, s in enumerate(scored[:50], 1):
        lines.append(f"| {i} | {s['Name']} | {s['Company']} | {s['Position']} | {s['Level']} |")
    lines.append('')

    # Contacts with emails
    with_email = [s for s in scored if s['Email'] and isinstance(s['Email'], str) and '@' in str(s['Email'])]
    lines.append(f'### Connections with Email Addresses ({len(with_email)} total)\n')
    if with_email:
        lines.append('| Name | Company | Position | Email |')
        lines.append('|------|---------|----------|-------|')
        for s in with_email[:100]:
            lines.append(f"| {s['Name']} | {s['Company']} | {s['Position']} | {s['Email']} |")
    lines.append('')

    return '\n'.join(lines)


# =========================================================================
# SECTION 3: COMPANY INTELLIGENCE
# =========================================================================
def company_intelligence(conn, apps, saved):
    lines = []
    lines.append('## 3. Company Intelligence\n')

    conn_by_company = defaultdict(list)
    for _, row in conn.iterrows():
        company = str(row.get('Company', '')).strip()
        if company and company != 'nan':
            conn_by_company[company.lower()].append({
                'Name': f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
                'Position': str(row.get('Position', '')),
                'Company': company,
            })

    # Applied companies
    app_companies = set()
    if 'Company Name' in apps.columns:
        app_companies = set(apps['Company Name'].dropna().str.strip().unique())
    elif len(apps.columns) > 0:
        # Try to find the right column
        for col in apps.columns:
            if 'company' in col.lower():
                app_companies = set(apps[col].dropna().str.strip().unique())
                break

    # Saved job companies
    saved_companies = set()
    if 'Company Name' in saved.columns:
        saved_companies = set(saved['Company Name'].dropna().str.strip().unique())
    elif len(saved.columns) > 0:
        for col in saved.columns:
            if 'company' in col.lower():
                saved_companies = set(saved[col].dropna().str.strip().unique())
                break

    # Cross-reference: applied + have connections
    lines.append(f'### Companies You Applied To ({len(app_companies)} total)\n')
    lines.append('#### You applied AND have connections here:\n')
    lines.append('| Company | Your Connections There |')
    lines.append('|---------|----------------------|')
    matches_app = 0
    for company in sorted(app_companies):
        key = company.lower()
        contacts = conn_by_company.get(key, [])
        if contacts:
            matches_app += 1
            names = ', '.join([f"{c['Name']} ({c['Position']})" for c in contacts[:5]])
            extra = f" +{len(contacts)-5} more" if len(contacts) > 5 else ""
            lines.append(f"| {company} | {names}{extra} |")
    lines.append(f'\n**{matches_app} of {len(app_companies)} applied companies have connections in your network.**\n')

    # Cross-reference: saved + have connections
    lines.append(f'### Companies from Saved Jobs ({len(saved_companies)} total)\n')
    lines.append('#### You saved a job AND have connections here:\n')
    lines.append('| Company | Your Connections There |')
    lines.append('|---------|----------------------|')
    matches_saved = 0
    for company in sorted(saved_companies):
        key = company.lower()
        contacts = conn_by_company.get(key, [])
        if contacts:
            matches_saved += 1
            names = ', '.join([f"{c['Name']} ({c['Position']})" for c in contacts[:5]])
            extra = f" +{len(contacts)-5} more" if len(contacts) > 5 else ""
            lines.append(f"| {company} | {names}{extra} |")
    lines.append(f'\n**{matches_saved} of {len(saved_companies)} saved-job companies have connections in your network.**\n')

    return '\n'.join(lines)


# =========================================================================
# SECTION 4: OUTREACH & RELATIONSHIP STRENGTH
# =========================================================================
def outreach_analysis(conn, msgs):
    lines = []
    lines.append('## 4. Outreach & Relationship Strength\n')

    # Build set of people Sidd has messaged with
    messaged_people = set()
    recent_messaged = set()

    sidd_variants = ['siddhant chauhan', 'sidd chauhan', 'siddhant']

    for _, row in msgs.iterrows():
        from_name = str(row.get('FROM', '')).strip().lower()
        to_name = str(row.get('TO', '')).strip().lower()
        date = parse_date_flexible(str(row.get('DATE', '')))

        # Identify the other person
        other = None
        if any(v in from_name for v in sidd_variants):
            other = str(row.get('TO', '')).strip()
        elif any(v in to_name for v in sidd_variants):
            other = str(row.get('FROM', '')).strip()

        if other:
            messaged_people.add(other.lower())
            if date and date >= NINETY_DAYS_AGO:
                recent_messaged.add(other.lower())

    # Classify connections
    active, warm, cold = [], [], []
    for _, row in conn.iterrows():
        name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
        name_lower = name.lower()
        score, level = seniority_score(row.get('Position'))
        entry = {
            'Name': name,
            'Company': str(row.get('Company', '')),
            'Position': str(row.get('Position', '')),
            'Score': score,
            'Level': level,
            'URL': str(row.get('URL', '')),
        }
        if name_lower in recent_messaged:
            active.append(entry)
        elif name_lower in messaged_people:
            warm.append(entry)
        else:
            cold.append(entry)

    lines.append(f'- **Active** (messaged in last 90 days): **{len(active)}**')
    lines.append(f'- **Warm** (messaged before, not recently): **{len(warm)}**')
    lines.append(f'- **Cold** (never messaged): **{len(cold)}**')
    lines.append('')

    # Priority outreach: high seniority + cold
    priority = sorted([c for c in cold if c['Score'] >= 2], key=lambda x: -x['Score'])
    lines.append(f'### Priority Outreach List — High-Seniority, Never Messaged ({len(priority)} contacts)\n')
    lines.append('| # | Name | Company | Position | Seniority |')
    lines.append('|---|------|---------|----------|-----------|')
    for i, p in enumerate(priority[:75], 1):
        lines.append(f"| {i} | {p['Name']} | {p['Company']} | {p['Position']} | {p['Level']} |")
    if len(priority) > 75:
        lines.append(f'\n*...and {len(priority) - 75} more. Full list in data.*\n')
    lines.append('')

    # Warm re-engagement: high seniority + warm
    warm_priority = sorted([c for c in warm if c['Score'] >= 2], key=lambda x: -x['Score'])
    lines.append(f'### Warm Re-Engagement — Messaged Before, High Seniority ({len(warm_priority)} contacts)\n')
    lines.append('| # | Name | Company | Position | Seniority |')
    lines.append('|---|------|---------|----------|-----------|')
    for i, p in enumerate(warm_priority[:50], 1):
        lines.append(f"| {i} | {p['Name']} | {p['Company']} | {p['Position']} | {p['Level']} |")
    lines.append('')

    return '\n'.join(lines)


# =========================================================================
# SECTION 5: NETWORK GAPS
# =========================================================================
def network_gaps(conn, apps, saved):
    lines = []
    lines.append('## 5. Network Gaps & Growth Patterns\n')

    conn_companies_lower = set(conn['Company'].dropna().str.strip().str.lower().unique())

    # Applied companies with NO connections
    app_companies = set()
    if 'Company Name' in apps.columns:
        app_companies = set(apps['Company Name'].dropna().str.strip().unique())
    elif len(apps.columns) > 0:
        for col in apps.columns:
            if 'company' in col.lower():
                app_companies = set(apps[col].dropna().str.strip().unique())
                break

    saved_companies = set()
    if 'Company Name' in saved.columns:
        saved_companies = set(saved['Company Name'].dropna().str.strip().unique())
    elif len(saved.columns) > 0:
        for col in saved.columns:
            if 'company' in col.lower():
                saved_companies = set(saved[col].dropna().str.strip().unique())
                break

    # Gaps: applied but no connections
    app_gaps = [c for c in sorted(app_companies) if c.lower() not in conn_companies_lower]
    lines.append(f'### Applied But NO Connections ({len(app_gaps)} companies)\n')
    lines.append('These are companies you applied to but have zero connections at — highest priority for network building:\n')
    for c in app_gaps[:50]:
        lines.append(f'- {c}')
    if len(app_gaps) > 50:
        lines.append(f'\n*...and {len(app_gaps) - 50} more.*')
    lines.append('')

    # Gaps: saved but no connections
    saved_gaps = [c for c in sorted(saved_companies) if c.lower() not in conn_companies_lower]
    lines.append(f'### Saved Jobs But NO Connections ({len(saved_gaps)} companies)\n')
    lines.append('Companies where you saved a job but have no one to reach out to:\n')
    for c in saved_gaps[:50]:
        lines.append(f'- {c}')
    if len(saved_gaps) > 50:
        lines.append(f'\n*...and {len(saved_gaps) - 50} more.*')
    lines.append('')

    return '\n'.join(lines)


# =========================================================================
# SECTION 6: REFERRAL PATH FINDER
# =========================================================================
def referral_paths(conn, apps, saved):
    lines = []
    lines.append('## 6. Referral Path Finder\n')

    # Build lookup
    conn_by_company = defaultdict(list)
    for _, row in conn.iterrows():
        company = str(row.get('Company', '')).strip()
        if company and company != 'nan':
            conn_by_company[company.lower()].append({
                'Name': f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
                'Position': str(row.get('Position', '')),
                'URL': str(row.get('URL', '')),
            })

    # Target companies = union of applied + saved
    target_companies = set()
    for df, col_hint in [(apps, 'Company Name'), (saved, 'Company Name')]:
        if col_hint in df.columns:
            target_companies.update(df[col_hint].dropna().str.strip().unique())
        elif len(df.columns) > 0:
            for col in df.columns:
                if 'company' in col.lower():
                    target_companies.update(df[col].dropna().str.strip().unique())
                    break

    lines.append(f'Target companies (from applications + saved jobs): **{len(target_companies)}**\n')
    lines.append('### Direct Connections at Target Companies\n')

    found = 0
    for company in sorted(target_companies):
        contacts = conn_by_company.get(company.lower(), [])
        if contacts:
            found += 1
            lines.append(f'**{company}** ({len(contacts)} connections):')
            for c in contacts:
                lines.append(f'  - {c["Name"]} — {c["Position"]}')
            lines.append('')

    lines.append(f'**You have direct connections at {found} of {len(target_companies)} target companies.**\n')

    return '\n'.join(lines)


# =========================================================================
# SECTION 7: MESSAGE INTELLIGENCE
# =========================================================================
def message_intelligence(msgs):
    lines = []
    lines.append('## 7. Message Intelligence\n')
    lines.append(f'**Total messages: {len(msgs)}**\n')

    sidd_variants = ['siddhant chauhan', 'sidd chauhan', 'siddhant']

    # Count messages per conversation partner
    partner_counts = Counter()
    partner_recent = {}

    for _, row in msgs.iterrows():
        from_name = str(row.get('FROM', '')).strip()
        to_name = str(row.get('TO', '')).strip()
        date = parse_date_flexible(str(row.get('DATE', '')))

        other = None
        if any(v in from_name.lower() for v in sidd_variants):
            other = to_name
        elif any(v in to_name.lower() for v in sidd_variants):
            other = from_name

        if other:
            partner_counts[other] += 1
            if date:
                if other not in partner_recent or date > partner_recent[other]:
                    partner_recent[other] = date

    # Most active conversations
    lines.append('### Most Active Conversation Partners (Top 30)\n')
    lines.append('| # | Person | Messages | Last Message |')
    lines.append('|---|--------|----------|-------------|')
    for i, (person, count) in enumerate(partner_counts.most_common(30), 1):
        last = partner_recent.get(person, None)
        last_str = last.strftime('%Y-%m-%d') if last else 'Unknown'
        lines.append(f'| {i} | {person} | {count} | {last_str} |')
    lines.append('')

    # Recently active (last 30 days)
    thirty_days = TODAY - timedelta(days=30)
    recent_partners = [(p, d) for p, d in partner_recent.items() if d and d >= thirty_days]
    recent_partners.sort(key=lambda x: x[1], reverse=True)
    lines.append(f'### Recently Active Conversations (Last 30 Days) — {len(recent_partners)} people\n')
    lines.append('| Person | Last Message | Total Messages |')
    lines.append('|--------|-------------|----------------|')
    for person, date in recent_partners:
        lines.append(f'| {person} | {date.strftime("%Y-%m-%d")} | {partner_counts[person]} |')
    lines.append('')

    # Job-related keyword scan
    job_keywords = ['interview', 'offer', 'referral', 'hiring', 'position', 'opportunity',
                    'resume', 'application', 'recruiter', 'role', 'salary', 'compensation']
    lines.append('### Job-Related Conversations (keyword matches)\n')
    lines.append('| Person | Date | Keyword | Snippet |')
    lines.append('|--------|------|---------|---------|')
    job_matches = []
    for _, row in msgs.iterrows():
        content = str(row.get('CONTENT', ''))
        if not content or content == 'nan':
            continue
        content_lower = content.lower()
        matched_kw = [kw for kw in job_keywords if kw in content_lower]
        if matched_kw:
            from_name = str(row.get('FROM', '')).strip()
            date = str(row.get('DATE', ''))
            snippet = content[:100].replace('\n', ' ').replace('|', '/')
            job_matches.append((from_name, date, ', '.join(matched_kw), snippet))

    # Deduplicate and show most recent
    job_matches.sort(key=lambda x: x[1], reverse=True)
    for m in job_matches[:50]:
        lines.append(f'| {m[0]} | {m[1][:10]} | {m[2]} | {m[3]} |')
    if len(job_matches) > 50:
        lines.append(f'\n*...{len(job_matches)} total job-related messages found.*')
    lines.append('')

    return '\n'.join(lines)


# =========================================================================
# MAIN
# =========================================================================
def main():
    print('Loading data...')
    conn = load_connections()
    msgs = load_messages()
    inv = load_invitations()
    apps = load_job_applications()
    saved = load_saved_jobs()

    print(f'  Connections: {len(conn)}')
    print(f'  Messages: {len(msgs)}')
    print(f'  Invitations: {len(inv)}')
    print(f'  Job Applications: {len(apps)}')
    print(f'  Saved Jobs: {len(saved)}')

    print('\nAnalyzing...')
    report = []
    report.append('# LinkedIn Network Analysis Report')
    report.append(f'*Generated: {TODAY.strftime("%Y-%m-%d")}*\n')
    report.append(f'**Profile: Siddhant (Sidd) Chauhan**\n')

    report.append(network_overview(conn))
    print('  [1/7] Network Overview done')

    report.append(influence_scoring(conn))
    print('  [2/7] Influence Scoring done')

    report.append(company_intelligence(conn, apps, saved))
    print('  [3/7] Company Intelligence done')

    report.append(outreach_analysis(conn, msgs))
    print('  [4/7] Outreach Analysis done')

    report.append(network_gaps(conn, apps, saved))
    print('  [5/7] Network Gaps done')

    report.append(referral_paths(conn, apps, saved))
    print('  [6/7] Referral Paths done')

    report.append(message_intelligence(msgs))
    print('  [7/7] Message Intelligence done')

    # Write report
    output = '\n'.join(report)
    (BASE / 'report.md').write_text(output, encoding='utf-8')
    print(f'\nReport written to: {BASE / "report.md"}')
    print(f'Total length: {len(output):,} characters')


if __name__ == '__main__':
    main()
