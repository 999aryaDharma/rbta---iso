# Campus VPS live ingestion decision — 2026-09-28

The researcher selected the campus VPS as the deployment target. Development and
verification remain native on Lenovo Windows. Offline replay remains supported.
Existing ASUS/Docker deployment files are historical; selecting a VPS does not
select Docker or authorize changes on a remote host without an access path.

## Confirmed and pending

| Item | Decision / evidence |
| --- | --- |
| Deployment host | Campus VPS, explicitly selected by researcher |
| Wazuh location | Campus environment; same host vs separate host unconfirmed |
| VPS OS and resources | Awaiting researcher/admin |
| Private Indexer reachability | Awaiting researcher/admin; never expose 9200 publicly |
| Poll vs collector push | Pending actual permitted access |
| Development | Existing native Windows Python ML environment and dashboard toolchain |
| Research behavior | Shared core, seven features, frozen model, unchanged replay semantics |
| Coverage | Preserve full-retention reconciliation; no event-time cutoff or late drop |
| Notifications | Shadow mode first; no external dispatch authorized by this implementation |

## Intended topology

```text
Campus Wazuh -- authorized private source --> live writer on campus VPS
                                               |
                                      separate live state/evidence
                                               |
                                      read-only API access
                                               |
Historical files --> isolated replay runs --> dashboard with explicit context
```

Before implementing the final transport/deployment profile, confirm the OS,
whether Wazuh is on the same machine, and whether private HTTPS Indexer access is
available. Credentials stay outside Git and task reports. If polling is selected,
the read-only account must support both search and the index discovery actually
used by the client; a successful search alone is insufficient validation.

## Revised implementation sequence

1. **Durability and ingestion prerequisites:** reproduce checkpoint/evidence loss;
   commit snapshot, IDs, pending scoring, history, outbox and cursor atomically;
   preserve legacy state migration; validate polling integrity, TLS, transport
   fingerprint parity and serialization of service mutations.
2. **Transport and worker:** select poll/push after access confirmation; bounded
   shutdown/retry, interprocess single writer, persistent run identity and model
   artifact pinning. A thread lock alone does not enforce one process writer.
3. **Read-only API and status:** do not instantiate a writable live service in the
   API for poll mode; expose actual cycle success, failure, staleness and readiness
   independently from replay readiness.
4. **Dashboard coexistence:** explicit live/replay selection, isolated queries,
   provenance, exports and outbox; display unavailable/error/empty states honestly.
5. **Authorized shadow deployment:** actual Wazuh smoke test, resource/lag evidence,
   then optional notification delivery with run-scoped idempotency at the receiver.

Streaming pagination is deferred until ordered consumption and partial-cycle
recovery are tested. Existing polling still materializes the full candidate set.
Do not claim it has bounded memory or is ready for an unmeasured campus retention
volume. Do not introduce an event-time cutoff as a workaround.

Idle grace and quarantine are not enabled. Their coverage/finalization contracts
must be specified before changing the operational behavior.

Deployment is **BLOCKED_EXTERNAL** pending access details and an authorized real
smoke test. Local test success is not evidence of campus connectivity.
