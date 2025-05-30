# Cross-Reference Map

This document maps the relationships between all documentation files in the MEXC Trading System documentation, ensuring proper cross-referencing and navigation.

## Core Navigation Documents

- [README.md](../README.md) → Main entry point, links to all major sections
- [docs/index.md](./index.md) → Comprehensive documentation index by category
- [docs/documentation_guide.md](./documentation_guide.md) → Documentation standards and formatting

## Architecture Documents

- [architecture/modular_trading_system_architecture.md](./architecture/modular_trading_system_architecture.md)
  - References: [mexc_api_component_mapping.md](./architecture/mexc_api_component_mapping.md), [rust_architecture.md](./reference/rust_architecture.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [architecture/mexc_api_component_mapping.md](./architecture/mexc_api_component_mapping.md)
  - References: [modular_trading_system_architecture.md](./architecture/modular_trading_system_architecture.md), [mexc_api_report.md](./reference/mexc_api_report.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

## Component Documentation

- [components/rust/market_data_processor.md](./components/rust/market_data_processor.md)
  - References: [market_data_processor_implementation.md](./components/rust/market_data_processor_implementation.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [components/rust/market_data_processor_implementation.md](./components/rust/market_data_processor_implementation.md)
  - References: [interoperability_guidelines.md](./development/guidelines/interoperability_guidelines.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md), [market_data_processor.md](./components/rust/market_data_processor.md)

- [components/rust/order_execution_implementation.md](./components/rust/order_execution_implementation.md)
  - References: [interoperability_guidelines.md](./development/guidelines/interoperability_guidelines.md), [design_and_implementation.md](./development/implementation/design_and_implementation.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [components/nodejs/decision_service.md](./components/nodejs/decision_service.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [components/python/signal_generator.md](./components/python/signal_generator.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [components/frontend/dashboard.md](./components/frontend/dashboard.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

## Development Documents

- [development/benchmarking_toolkit.md](./development/benchmarking_toolkit.md)
  - References: [setup_guide.md](./development/setup/setup_guide.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [development/setup/setup_guide.md](./development/setup/setup_guide.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md), [benchmarking_toolkit.md](./development/benchmarking_toolkit.md)

- [development/guidelines/interoperability_guidelines.md](./development/guidelines/interoperability_guidelines.md)
  - Referenced by: [market_data_processor_implementation.md](./components/rust/market_data_processor_implementation.md), [order_execution_implementation.md](./components/rust/order_execution_implementation.md), [README.md](../README.md), [index.md](./index.md)

- [development/guidelines/llm_developer_guidelines.md](./development/guidelines/llm_developer_guidelines.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [development/implementation/design_and_implementation.md](./development/implementation/design_and_implementation.md)
  - References: [order_execution_implementation.md](./components/rust/order_execution_implementation.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

## Operations Documents

- [operations/monitoring/grafana_dashboard_template.md](./operations/monitoring/grafana_dashboard_template.md)
  - References: [setup_guide.md](./development/setup/setup_guide.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

## Reference Documents

- [reference/mexc_api_report.md](./reference/mexc_api_report.md)
  - References: [mexc_api_component_mapping.md](./architecture/mexc_api_component_mapping.md)
  - Referenced by: [README.md](../README.md), [index.md](./index.md)

- [reference/implementation_priorities.md](./reference/implementation_priorities.md)
  - References: [architecture_and_frameworks.md](./reference/architecture_and_frameworks.md)
  - Referenced by: [index.md](./index.md)

- [reference/architecture_and_frameworks.md](./reference/architecture_and_frameworks.md)
  - References: [implementation_priorities.md](./reference/implementation_priorities.md), [modular_trading_system_architecture.md](./architecture/modular_trading_system_architecture.md)
  - Referenced by: [index.md](./index.md)

- [reference/development_documentation.md](./reference/development_documentation.md)
  - Referenced by: [index.md](./index.md)

## Cross-Reference Implementation

To ensure proper cross-referencing, each document should:

1. Include a "Related Documents" section at the end
2. Use relative links to reference other documents
3. Provide context for why the reader might want to consult the referenced document

Example format for "Related Documents" section:

```markdown
## Related Documents

- [Document Name](./path/to/document.md) - Brief description of relevance
- [Another Document](./path/to/another.md) - Why this document is related
```

This cross-reference map should be maintained as documentation evolves to ensure navigation remains clear and comprehensive.
