# LLM Developer Guidelines and Permission Controls

## Overview

This document outlines strict guidelines and permission controls for LLM developers working on the MEXC trading system. These guidelines are designed to ensure that development follows the approved implementation plan, with no unauthorized features or deviations.

## Core Principles

1. **Explicit Permission Required**
   - No feature, component, or functionality may be implemented without explicit approval
   - All development must strictly follow the approved implementation plan
   - Any deviation, no matter how small, requires prior written approval

2. **Incremental Development**
   - Development must proceed in the exact order specified in the implementation plan
   - Each step must be completed and approved before proceeding to the next
   - No skipping ahead or implementing features from future phases

3. **Documentation First**
   - All features must be documented before implementation
   - Documentation must be approved before coding begins
   - Changes to documentation require approval

4. **Transparency**
   - All code changes must be visible and reviewable
   - Regular progress reports are required
   - Any challenges or blockers must be reported immediately

## Permission Workflow

### 1. Feature Implementation Request

Before implementing any feature, the LLM developer must submit a formal request containing:

- Specific reference to the implementation plan step
- Detailed description of the proposed implementation
- List of components and files to be modified
- External dependencies to be added (if any)
- Estimated time for implementation

**Template**:
```
FEATURE IMPLEMENTATION REQUEST

Reference: [Phase X, Step Y: Feature Name]
Description: [Detailed description of implementation approach]
Components: [List of components to be created/modified]
Dependencies: [List of external libraries/dependencies]
Estimated Time: [X hours/days]

Additional Notes: [Any clarifications or questions]
```

### 2. Approval Process

The user must explicitly approve each feature implementation request before work begins:

- All requests must be submitted in writing
- Approval must be explicit and in writing
- Partial approvals must clearly state what is approved and what is not
- Approval is valid only for the specific implementation described

### 3. Implementation Boundaries

Once approved, the LLM developer:

- May only implement what was explicitly approved
- Must follow the approved approach exactly
- May not add "nice-to-have" features or enhancements
- Must stop and request approval for any necessary deviations

### 4. Review and Verification

After implementation:

- All code must be submitted for review
- Implementation must be verified against the original request
- Any discrepancies must be explained and approved
- User must explicitly approve the implementation before it is considered complete

## Specific Restrictions

### Code Modifications

1. **File Structure**
   - Must follow the exact project structure defined in the implementation plan
   - No creation of additional directories or files without approval
   - No reorganization of existing files without approval

2. **Dependencies**
   - No new dependencies may be added without explicit approval
   - Version changes to approved dependencies require approval
   - All dependencies must be from approved sources

3. **Code Style**
   - Must follow the defined coding standards
   - No introduction of new patterns or approaches
   - Must use approved frameworks and libraries as specified

### Feature Implementation

1. **Scope Limitations**
   - Implement only what is explicitly described in the plan
   - No "future-proofing" or implementing ahead of schedule
   - No experimental features or "explorations"

2. **UI/UX Elements**
   - Must follow the approved design system
   - No additional UI elements without approval
   - No modifications to approved UI patterns

3. **Data Handling**
   - Must use only approved data sources
   - No additional data collection without approval
   - No changes to data models without approval

### Security and Performance

1. **Security Measures**
   - Must implement all specified security measures
   - No weakening of security for convenience
   - No security-related changes without approval

2. **Performance Optimizations**
   - Must follow approved optimization strategies
   - No premature optimization
   - Performance changes must be measurable and approved

## Enforcement Mechanisms

### 1. Code Review Gates

All code must pass through mandatory review gates:

- **Pre-implementation Review**: Approval of approach before coding
- **Implementation Review**: Review of actual code changes
- **Post-implementation Review**: Verification that implementation matches approval

### 2. Automated Checks

Automated systems will enforce:

- Dependency compliance
- Code style and standards
- Test coverage requirements
- Documentation requirements

### 3. Regular Audits

Regular audits will be conducted to ensure:

- All implemented features were properly approved
- No unauthorized features exist in the codebase
- All documentation is current and accurate

### 4. Versioned Approvals

All approvals will be:

- Documented with timestamps
- Associated with specific versions of the implementation plan
- Tracked in a centralized approval registry

## Communication Protocol

### 1. Regular Check-ins

The LLM developer must provide:

- Daily progress updates
- Weekly summary reports
- Phase completion reports

### 2. Issue Reporting

Any issues must be reported immediately:

- Technical blockers
- Specification ambiguities
- Conflicts between requirements
- Discovered security concerns

### 3. Change Requests

For any needed changes to the approved plan:

- Submit formal change request
- Provide justification for the change
- Wait for explicit approval before proceeding

## Documentation Requirements

### 1. Implementation Documentation

For each implemented feature:

- Technical documentation describing the implementation
- Usage examples and API references
- Testing approach and results
- Known limitations or issues

### 2. Decision Records

For all significant decisions:

- Document the context and problem
- List alternatives considered
- Explain the chosen solution
- Record approval of the decision

### 3. Progress Tracking

Maintain up-to-date documentation of:

- Completed features
- In-progress work
- Pending approvals
- Upcoming features

## Compliance Statement

The LLM developer must acknowledge and agree to these guidelines before beginning work:

```
I acknowledge that I have read and understood the LLM Developer Guidelines and Permission Controls. I agree to strictly follow these guidelines and understand that any deviation without explicit approval is prohibited. I will not implement any features, components, or functionality that has not been explicitly approved, regardless of how beneficial or minor they may seem.

I understand that my role is to implement exactly what has been approved in the implementation plan, nothing more and nothing less, and that all changes require explicit permission.
```

## Conclusion

These guidelines establish a strict permission-based development process that ensures the MEXC trading system is built exactly according to the approved implementation plan. By following these guidelines, the LLM developer will help create a system that meets the user's requirements without unauthorized features or scope creep.

The user maintains complete control over the development process, with explicit approval required at every step. This ensures transparency, accountability, and alignment with the user's vision for the system.
