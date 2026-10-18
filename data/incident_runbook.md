# Incident Response Runbook

## Triage
SEV1 means customer-facing outage or data risk. The first responder pages the on-call engineer, opens an incident channel, and posts the known symptoms within fifteen minutes.

## Database failover
If the primary database fails, promote the read replica to primary. First confirm replication lag is under five seconds, then run the failover checklist, point the application at the promoted replica, and watch write latency for fifteen minutes. Never fail back during an active incident; schedule the failback for the next business day.

## Communication
Post status updates every thirty minutes for SEV1 incidents. Each update states what changed, what is being tried next, and when the next update will arrive.

## Post-incident review
Within three business days, write a blameless review with timeline, root cause, customer impact, and follow-up actions with owners.
