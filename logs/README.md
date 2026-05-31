# Logs

This directory contains application and experiment execution logs.

## Purpose

- Track application execution flow
- Debug issues and troubleshoot problems
- Monitor long-running experiments
- Maintain audit trail of all operations

## Log Files

- `benchmarks.log`: Main application log
- `experiments_{date}.log`: Experiment-specific logs
- `errors.log`: Error messages and stack traces

## Log Format

```
TIMESTAMP - MODULE - LEVEL - MESSAGE
2024-01-15 10:30:45 - experiments.benchmark - INFO - Starting experiment: baseline_comparison
2024-01-15 10:30:46 - datasets.loaders - INFO - Loading dataset: iris
```

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations
- **CRITICAL**: Critical errors requiring immediate attention

## Retention

Log files are rotated monthly. Archive old logs if needed for long-term benchmarking studies.
