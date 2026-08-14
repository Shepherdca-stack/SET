# Supporting numeric scripts T4-T12

These nine versioned scripts carry the full per-check numeric detail behind derivation chains 26 through 32. Their conclusions are absorbed into the master state and workbook, and their load-bearing results are condensed into the 94-check verification suite at the repository root, so the suite alone reproduces the record's verification status. The scripts are retained for full transparency: they show every intermediate quantity, grid, and convergence test behind the condensed checks.

Expected manifest:

| File | Chain | Content |
|---|---|---|
| SET_time_T4_wilson.py | 26 | Wilson cancellation, alternating-sum bound, resonance comb, thermalization bar |
| SET_time_T5_relic.py | 27 | Relic exclusion at the 469 MeV window, three-way overdetermination |
| SET_time_T6_e80_audit.py | 28 | E80 unit audit, tick selection, candidate B kill, candidate A upgrade |
| SET_time_T7_coupling.py | 29 | Coupling derivation, WKB pinch quarantine, gravitational ceiling |
| SET_time_T8_carrier.py | 30 | Inventory carrier elimination, KK graviton lifetime kill |
| SET_time_T9_dust.py | 31 | Geometric dust candidate, cold/collisionless/stability checks |
| SET_time_T10_partition.py | 31 | Partition no-go theorem E114, P-RING reduction, cycloid rival disclosure |
| SET_time_T11_gradient.py | 32 | Motion law E115, gradient no-go, integrable triple E116 |
| SET_time_T12_landau.py | 32 | Q-LAW quantization, Landau ladder, E117 comb identity |

All nine scripts are present and were run in full on August 12, 2026, under the pinned environment in the repository's requirements.txt (numpy 2.4.4, scipy 1.17.1, sympy 1.14.0, Python 3.12). Every script completed with zero FAIL lines and reached its "All committed checks resolved" verdict. Committed check counts per run: T4: 8, T5: 5, T6: 5, T7: 5, T8: 3, T9: 10, T10: 14, T11: 11, T12: 9, total 70, alongside the scripts' [OBS] observation lines, which are logged findings rather than pass/fail checks. Two session-log lines in the master state record T11 as 12 checks and T12 as 10; the scripts themselves contain 11 and 9 check calls respectively, the runs are complete, and the uploaded scripts govern. The count correction is queued for the next registry merge.

Provenance note carried from the master state: T4 was recovered verbatim from the session-1 transcript and rerun green; session-1 scripts T1-T3 are absent from the archive, and their PROVEN conclusions are absorbed into the master state with that provenance flagged.
