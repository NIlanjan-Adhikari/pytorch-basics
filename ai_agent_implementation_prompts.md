# AI Agent Implementation Prompts for Copilot

## System Overview Prompt

You are tasked with building an AI agent that automatically reads business requirements from Excel files and generates complex YAML configuration files for LangGraph-based document extraction workflows. The agent must understand field mappings, validation rules, workflow specifications, and translate these into production-ready YAML configurations.

## Core Implementation Requirements

### 1. Excel Parser Component

```
Create a robust Excel parsing module that:

**Primary Function**: Parse business requirements from Excel files containing field specifications, data types, validation rules, and workflow definitions.

**Technical Requirements**:
- Use openpyxl library with read_only=True and data_only=True for performance
- Support .xlsx and .xlsm file formats
- Handle multiple sheets with different purposes (field_mappings, validation_rules, workflow_specs)
- Extract structured data including:
  * Field names and descriptions
  * Data types (int, str, date, etc.)
  * Validation requirements
  * Translation needs
  * Workflow node specifications
  * Edge definitions and routing logic

**Expected Input**: Excel file with sheets named like "Field_Mapping", "Validation_Rules", "Workflow_Specifications"
**Expected Output**: Structured dictionary with parsed requirements

**Error Handling**: 
- Graceful handling of missing sheets
- Validation of required columns
- Memory-efficient processing for large files
- Detailed error reporting for malformed data

**Code Structure**: Create a class `ExcelProcessor` with methods for parsing different sheet types.
```

### 2. LangGraph Workflow Orchestrator

```
Build a LangGraph-based workflow orchestration system that:

**Primary Function**: Coordinate the entire YAML generation process using stateful agent workflows.

**Technical Requirements**:
- Use LangGraph's StateGraph for workflow management
- Implement specialized nodes for: excel_parsing, llm_analysis, template_selection, yaml_generation, validation, refinement
- Add conditional routing based on validation results
- Include human-in-the-loop capabilities for manual review
- Support checkpointing and error recovery

**State Management**: Define a TypedDict called `ConfigGenerationState` with fields:
- excel_file: str
- parsed_data: Dict[str, Any]
- requirements: str
- template_config: Dict[str, Any]
- generated_yaml: str
- validation_results: List[str]
- confidence_scores: Dict[str, float]

**Workflow Logic**: 
- Sequential processing through nodes
- Quality gates that route to refinement or manual review on failures
- Automatic retry logic with exponential backoff

**Code Structure**: Create a function `create_config_generation_workflow()` that returns a compiled LangGraph workflow.
```

### 3. Template Engine with LLM Enhancement

```
Develop a hybrid YAML generation system that combines Jinja2 templates with LLM enhancement:

**Primary Function**: Generate YAML configurations using base templates enhanced by LLM for complex business logic.

**Technical Requirements**:
- Use Jinja2 for base template structure
- Use ruamel.yaml for YAML formatting and comment preservation
- Integrate GPT-4 for complex logic enhancement
- Support multiple template types (langgraph_workflow, doc_extraction, validation_schemas)
- Custom Jinja2 filters for YAML-safe output

**Template Structure**: Create base templates for:
- LangGraph workflow definitions
- Sub-model configurations
- Field mapping specifications
- Edge and node definitions

**LLM Enhancement**: Use LLM to:
- Generate complex conditional logic
- Create dynamic field mappings
- Add error handling patterns
- Optimize workflow performance

**Expected Input**: Parsed Excel data and template type
**Expected Output**: Valid YAML configuration string

**Code Structure**: Create a class `YAMLConfigGenerator` with template rendering and LLM enhancement methods.
```

### 4. Multi-Layer Validation System

```
Implement a comprehensive validation framework with four validation layers:

**Primary Function**: Ensure generated YAML configurations are syntactically correct, schema-compliant, and meet business requirements.

**Validation Layers**:
1. **Syntax Validation**: YAML syntax correctness
2. **Schema Validation**: Compliance with predefined schemas using Cerberus
3. **Business Rules Validation**: Custom business logic validation
4. **LangGraph-Specific Validation**: Workflow connectivity and node reference validation

**Technical Requirements**:
- Progressive validation that stops on critical failures
- Detailed error reporting with line numbers and suggestions
- Confidence scoring based on validation results
- Automatic recommendation generation for fixes

**Schema Definitions**: Create validation schemas for:
- LangGraph workflow structure
- Sub-model configurations
- Field specifications
- Node and edge definitions

**Expected Input**: YAML configuration string
**Expected Output**: Validation results with pass/fail status, errors, and recommendations

**Code Structure**: Create a class `ConfigurationValidator` with methods for each validation layer.
```

### 5. Integration Module

```
Build an integration system that safely merges generated configurations with existing systems:

**Primary Function**: Enable safe deployment of new configurations without breaking existing workflows.

**Integration Strategies**:
- **Incremental Migration**: Gradually replace configuration components
- **Blue-Green Deployment**: Full environment switching
- **Canary Rollout**: Percentage-based traffic routing

**Technical Requirements**:
- Compatibility analysis between old and new configurations
- Breaking change detection and handling
- Rollback capabilities
- Migration planning and execution
- Integration testing hooks

**Safety Features**:
- Dry-run mode for testing changes
- Configuration backup and restore
- Dependency analysis
- Impact assessment reporting

**Code Structure**: Create a class `LangGraphSystemIntegrator` with migration strategy implementations.
```

### 6. Quality Assurance Pipeline

```
Create a comprehensive QA system with multiple testing frameworks:

**Primary Function**: Ensure generated configurations meet quality standards before deployment.

**Testing Frameworks**:
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end workflow testing
- **Property-Based Tests**: Configuration invariant testing
- **Performance Tests**: Load and stress testing

**Quality Metrics**:
- Configuration validation pass rates
- Processing time benchmarks
- Memory usage monitoring
- Error rate tracking

**Automated Testing**: 
- Continuous integration hooks
- Automated test report generation
- Performance regression detection
- Quality gate enforcement

**Code Structure**: Create a class `ConfigurationQA` with methods for each testing framework.
```

## Implementation Prompts by File

### File 1: `excel_parser.py`

```
Create a production-ready Excel parser with the following specifications:

**Class**: `ExcelProcessor`

**Methods to implement**:
- `parse_requirements_excel(file_path: str) -> Dict[str, Any]`
- `_extract_field_mappings(sheet) -> List[Dict[str, str]]`
- `_extract_validation_rules(sheet) -> Dict[str, Any]`
- `_extract_workflow_specs(sheet) -> Dict[str, Any]`
- `validate_excel_structure(file_path: str) -> bool`

**Key Features**:
- Memory-efficient parsing with openpyxl
- Support for multiple sheet types
- Error handling and logging
- Data validation and sanitization
- Caching for repeated processing

**Performance Requirements**: Handle 10,000+ rows in under 30 seconds
```

### File 2: `workflow_orchestrator.py`

```
Build a LangGraph workflow orchestrator with these specifications:

**Main Function**: `create_config_generation_workflow()`

**Node Functions to implement**:
- `excel_parsing_node(state: ConfigGenerationState) -> ConfigGenerationState`
- `llm_analysis_node(state: ConfigGenerationState) -> ConfigGenerationState`
- `template_selection_node(state: ConfigGenerationState) -> ConfigGenerationState`
- `yaml_generation_node(state: ConfigGenerationState) -> ConfigGenerationState`
- `validation_node(state: ConfigGenerationState) -> ConfigGenerationState`
- `refinement_node(state: ConfigGenerationState) -> ConfigGenerationState`

**Routing Logic**: `quality_gate_function(state: ConfigGenerationState) -> str`

**Error Handling**: Implement retry logic and fallback strategies
**Monitoring**: Add logging and metrics collection
```

### File 3: `yaml_generator.py`

```
Create a hybrid YAML generator with these specifications:

**Class**: `YAMLConfigGenerator`

**Methods to implement**:
- `generate_config(requirements: Dict, template_type: str) -> str`
- `_enhance_with_llm(base_config: str, requirements: Dict) -> str`
- `_validate_and_refine(config: str) -> str`
- `_yaml_safe_filter(value: Any) -> str`

**Template Files to create**:
- `langgraph_workflow_base.j2`
- `sub_models_template.j2`
- `validation_schema_template.j2`

**LLM Integration**: Use OpenAI GPT-4 with specific prompts for YAML enhancement
**Performance Target**: Sub-5-minute generation for complex workflows
```

### File 4: `validator.py`

```
Implement a multi-layer validation system with these specifications:

**Class**: `ConfigurationValidator`

**Methods to implement**:
- `validate_configuration(yaml_content: str) -> Dict[str, Any]`
- `_validate_yaml_syntax(content: str) -> Dict[str, Any]`
- `_validate_schema_compliance(config: Dict) -> Dict[str, Any]`
- `_validate_business_rules(config: Dict) -> Dict[str, Any]`
- `_validate_langgraph_specific(config: Dict) -> Dict[str, Any]`
- `_calculate_confidence(layer_results: Dict) -> float`

**Schema Files to create**:
- `langgraph_workflow_schema.json`
- `sub_model_schema.json`
- `field_mapping_schema.json`

**Validation Target**: 99.9% accuracy with detailed error reporting
```

### File 5: `integrator.py`

```
Build a system integration module with these specifications:

**Class**: `LangGraphSystemIntegrator`

**Methods to implement**:
- `integrate_generated_config(config: str, strategy: str) -> Dict[str, Any]`
- `_analyze_compatibility(old_config: Dict, new_config: Dict) -> Dict[str, Any]`
- `_incremental_migration(new_config: Dict) -> Dict[str, Any]`
- `_blue_green_deployment(new_config: Dict) -> Dict[str, Any]`
- `_canary_rollout(new_config: Dict) -> Dict[str, Any]`
- `_handle_breaking_changes(compatibility: Dict, new_config: Dict) -> Dict[str, Any]`

**Safety Features**: Include rollback capabilities and change impact analysis
**Testing**: Support dry-run mode for safe testing
```

### File 6: `qa_pipeline.py`

```
Create a quality assurance pipeline with these specifications:

**Class**: `ConfigurationQA`

**Methods to implement**:
- `quality_assurance_pipeline(config: str) -> Dict[str, Any]`
- `_run_unit_tests(config: str) -> Dict[str, Any]`
- `_run_integration_tests(config: str) -> Dict[str, Any]`
- `_run_e2e_tests(config: str) -> Dict[str, Any]`
- `_run_property_tests(config: str) -> Dict[str, Any]`
- `_performance_testing(config: str) -> Dict[str, Any]`
- `_assess_deployment_readiness(test_results: Dict, perf_metrics: Dict) -> bool`

**Testing Frameworks**: Integrate pytest, hypothesis for property-based testing
**Performance Monitoring**: Include resource usage tracking
```

## Success Criteria

The implemented system should achieve:

- **Accuracy**: >95% configuration validation pass rate
- **Performance**: <5 minutes end-to-end processing time
- **Reliability**: <2% configuration-related incident rate  
- **Efficiency**: 50% reduction in manual configuration overhead

## Testing Requirements

Create comprehensive tests for:
- Unit tests for each component (>90% code coverage)
- Integration tests for workflow orchestration
- End-to-end tests with sample Excel files
- Performance benchmarks with large datasets
- Error handling and edge case validation

## Documentation Requirements

Generate documentation including:
- API documentation for all classes and methods
- Configuration schema documentation
- Usage examples and tutorials
- Troubleshooting guides
- Performance optimization tips

## Deployment Configuration

Create deployment files:
- `pyproject.toml` with uv dependency management
- `docker-compose.yml` for containerized deployment
- `langgraph.json` for LangGraph Platform deployment
- Environment configuration templates
- CI/CD pipeline configuration with uv installation

Use these prompts individually or in combination to guide the implementation of each component. Each prompt is designed to be specific enough for an AI coding assistant to generate working code while maintaining the overall system architecture.