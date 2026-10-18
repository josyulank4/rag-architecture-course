# Deployment Policy

## Release windows
Production deployments happen on Tuesdays and Thursdays between 10:00 and 16:00 UTC. Emergency hotfix deployments may run outside the release windows only with incident commander approval and a written note in the change log.

## Approvals
Every production deployment requires two approvals: one from the service owner and one from the release manager. The deploy pipeline records both approvals before the deploy button is enabled.

## Rollback procedure
If the error rate exceeds 2 percent for five minutes after a deployment, roll back immediately. Run `shipctl rollback --env prod --to previous`, confirm the previous version is serving traffic, and check the health dashboard for ten minutes. The rollback is complete only when the error rate returns below 0.5 percent and the on-call engineer signs the incident note.

## Change log
After every deployment or rollback, record the version, approvers, start time, end time, and observed error rate in the deployment change log.
