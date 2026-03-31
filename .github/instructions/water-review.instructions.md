---
applyTo: "water/**"
excludeAgent: "coding-agent"
---

- Never make comments that some code may not compile, we have CI that proves it does or does not.
- Consider whether the proposed code integrates into the overall architecture of the project.
- Suggest using helper functions already available in the project, potentially by refactoring or generalizing them.
- Follow LLVM coding style.
- Suggest using helper functions from LLVM libraries, e.g., `llvm::join`, or methods of LLVM data structures when possible.
- Double-check that when logic of a function changes, the documentation is updated accordingly and so are the tests, if present.
- Ensure all top-level functions, classes and variables have documentation. We prefer documenting in headers. Each function should only have one documentation block, i.e., avoid having documentation on both the declaration and the definition.
- Check that there is reasonable test coverage of the new or changed functionality. However, we should not trigger assertions in tests, those should not be possible on normal flow.
