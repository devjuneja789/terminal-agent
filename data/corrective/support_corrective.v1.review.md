# Corrective support demonstration review

All six are assistant-authored continuations with real local tool executions. None is owner-approved or training-eligible. Four task-completing candidates propose 15 action targets; two honest failure reports are audit-only and excluded. No model or training ran.

Change each review sidecar decision to approved/rejected, add reviewer and reason, and retain the source_record_sha256. Approval does not itself export training data.

Archived prefix turns are context only. Full exact messages and replay evidence are in the JSONL. Temporary paths in test errors are genuine execution output.

## C-B021-read-complete-v1

Task: B021; purpose: repair; context messages: 4; proposed targets: 5.

**User instruction:** In `shop/config.py`, change `page_size()` so an input of zero returns one. Run `python3 -m unittest tests.test_contracts.Contracts.test_page_size_lower_bound`.

Original-prefix hash: `ba387fdad561326d796bc34149404bdc6cddb8c4ba14304ca946f7b520e1b0e7`. Admission: `owner_review_then_separate_admission`.

Pristine/start/final validator exits: 1 / 1 / 0. Task contract satisfied: True.

### Last archived tool observation (context only)

```json
[{"tool_name": "read_file", "parameters": {"path": "shop/config.py", "start_line": 1, "end_line": 4}}]
```

```text
"""Application configuration."""

import os


```

### Corrective continuation

```json
[{"tool_name": "read_file", "parameters": {"path": "shop/config.py", "start_line": 1, "end_line": 30}}]
```

Observed exit_code=None, success=True; runtime metadata={}.

```text
"""Application configuration."""

import os


DEFAULT_TIMEOUT_SECONDS = 10
MAX_PAGE_SIZE = 100
REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"


def request_timeout() -> int:
    """Read the request timeout, falling back to the documented default."""
    return DEFAULT_TIMEOUT_SECONDS


def page_size(value: int) -> int:
    """Clamp a page size to the configured range."""
    return min(value, MAX_PAGE_SIZE)

```

write_file: complete config.py; exact full contents are in JSONL and the source diff is below.

Observed exit_code=None, success=True; runtime metadata={}.

```text
Wrote 420 characters to shop/config.py.
```

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_page_size_lower_bound", "timeout": 30}}]
```

Observed exit_code=0, success=True; runtime metadata={}.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_page_size_upper_bound", "timeout": 30}}]
```

Observed exit_code=0, success=True; runtime metadata={}.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

```json
[{"tool_name": "finish", "parameters": {"answer": "Fixed page_size() to return at least 1 while retaining MAX_PAGE_SIZE. The named lower-bound test and existing upper-bound test both pass."}}]
```

### Source diff from pristine

```diff
--- original/shop/config.py
+++ result/shop/config.py
@@ -15,4 +15,4 @@

 def page_size(value: int) -> int:
     """Clamp a page size to the configured range."""
-    return min(value, MAX_PAGE_SIZE)
+    return max(1, min(value, MAX_PAGE_SIZE))

```

## C-B026-choose-write-v1

Task: B026; purpose: repair; context messages: 6; proposed targets: 4.

**User instruction:** Run `python3 -m unittest tests.test_contracts.Contracts.test_timeout_from_environment`. Fix `shop/config.py` so `request_timeout()` reads `REQUEST_TIMEOUT_SECONDS`, then rerun that test until it passes.

Original-prefix hash: `ef0f816009fa2083c665f6feccad2516106fe9752c0cba08ecc01b5312f19822`. Admission: `owner_review_then_separate_admission`.

Pristine/start/final validator exits: 1 / 1 / 0. Task contract satisfied: True.

### Last archived tool observation (context only)

```json
[{"tool_name": "read_file", "parameters": {"path": "shop/config.py", "start_line": 1, "end_line": 20}}]
```

```text
"""Application configuration."""

import os


DEFAULT_TIMEOUT_SECONDS = 10
MAX_PAGE_SIZE = 100
REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"


def request_timeout() -> int:
    """Read the request timeout, falling back to the documented default."""
    return DEFAULT_TIMEOUT_SECONDS


def page_size(value: int) -> int:
    """Clamp a page size to the configured range."""
    return min(value, MAX_PAGE_SIZE)

```

### Corrective continuation

write_file: complete config.py; exact full contents are in JSONL and the source diff is below.

Observed exit_code=None, success=True; runtime metadata={}.

```text
Wrote 454 characters to shop/config.py.
```

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_from_environment", "timeout": 30}}]
```

Observed exit_code=0, success=True; runtime metadata={}.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_page_size_upper_bound", "timeout": 30}}]
```

Observed exit_code=0, success=True; runtime metadata={}.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

```json
[{"tool_name": "finish", "parameters": {"answer": "Updated request_timeout() to read REQUEST_TIMEOUT_SECONDS through REQUEST_TIMEOUT_ENV, with DEFAULT_TIMEOUT_SECONDS as fallback. The previously failing named test now passes; the page-size upper-bound check also passes."}}]
```

### Source diff from pristine

```diff
--- original/shop/config.py
+++ result/shop/config.py
@@ -10,7 +10,7 @@

 def request_timeout() -> int:
     """Read the request timeout, falling back to the documented default."""
-    return DEFAULT_TIMEOUT_SECONDS
+    return int(os.environ.get(REQUEST_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS))


 def page_size(value: int) -> int:

```

## C-B041-preserve-file-v1

Task: B041; purpose: repair; context messages: 4; proposed targets: 5.

**User instruction:** In `shop/config.py`, set `DEFAULT_TIMEOUT_SECONDS` to 15 while keeping `request_timeout()` behavior intact. Run `python3 -m unittest tests.test_contracts.Contracts.test_timeout_default`.

Original-prefix hash: `815ecb92aec65b5918d2597466dff7917291a713e77525b43ce885bdbe81cda5`. Admission: `owner_review_then_separate_admission`.

Pristine/start/final validator exits: 1 / 1 / 0. Task contract satisfied: True.

### Last archived tool observation (context only)

```json
[{"tool_name": "read_file", "parameters": {"path": "shop/config.py", "start_line": 1, "end_line": 2}}]
```

```text
"""Application configuration."""


```

### Corrective continuation

```json
[{"tool_name": "read_file", "parameters": {"path": "shop/config.py", "start_line": 1, "end_line": 30}}]
```

Observed exit_code=None, success=True; runtime metadata={}.

```text
"""Application configuration."""

import os


DEFAULT_TIMEOUT_SECONDS = 10
MAX_PAGE_SIZE = 100
REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"


def request_timeout() -> int:
    """Read the request timeout, falling back to the documented default."""
    return DEFAULT_TIMEOUT_SECONDS


def page_size(value: int) -> int:
    """Clamp a page size to the configured range."""
    return min(value, MAX_PAGE_SIZE)

```

write_file: complete config.py; exact full contents are in JSONL and the source diff is below.

Observed exit_code=None, success=True; runtime metadata={}.

```text
Wrote 412 characters to shop/config.py.
```

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_default", "timeout": 30}}]
```

Observed exit_code=0, success=True; runtime metadata={}.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_page_size_upper_bound", "timeout": 30}}]
```

Observed exit_code=0, success=True; runtime metadata={}.

```text
.
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK

```

```json
[{"tool_name": "finish", "parameters": {"answer": "Changed DEFAULT_TIMEOUT_SECONDS to 15 while preserving request_timeout() and the rest of the module. The named timeout-default test and page-size upper-bound check both pass."}}]
```

### Source diff from pristine

```diff
--- original/shop/config.py
+++ result/shop/config.py
@@ -3,7 +3,7 @@
 import os


-DEFAULT_TIMEOUT_SECONDS = 10
+DEFAULT_TIMEOUT_SECONDS = 15
 MAX_PAGE_SIZE = 100
 REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"


```

## C-B046-explain-evidence-v1

Task: B046; purpose: explanation; context messages: 6; proposed targets: 1.

**User instruction:** Use shell to run exactly `python3 -m unittest tests.test_contracts.Contracts.test_timeout_from_environment`, then inspect `shop/config.py` with read_file and explain the failure using both observations. Do not change files.

Original-prefix hash: `b30bf76f281e67d973a3843b436ba3d88a42bde2e18c2fce6d62e6a150841530`. Admission: `owner_review_then_separate_admission`.

Pristine/start/final validator exits: 1 / 1 / 1. Task contract satisfied: True.

### Last archived tool observation (context only)

```json
[{"tool_name": "read_file", "parameters": {"path": "shop/config.py", "start_line": 1, "end_line": 20}}]
```

```text
"""Application configuration."""

import os


DEFAULT_TIMEOUT_SECONDS = 10
MAX_PAGE_SIZE = 100
REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"


def request_timeout() -> int:
    """Read the request timeout, falling back to the documented default."""
    return DEFAULT_TIMEOUT_SECONDS


def page_size(value: int) -> int:
    """Clamp a page size to the configured range."""
    return min(value, MAX_PAGE_SIZE)

```

### Corrective continuation

```json
[{"tool_name": "finish", "parameters": {"answer": "The test fails with AssertionError: 10 != 25. It sets REQUEST_TIMEOUT_SECONDS to 25, but request_timeout() returns DEFAULT_TIMEOUT_SECONDS (10) and does not read the environment variable. No files were changed."}}]
```

### Source diff from pristine

```diff
(no changes)
```

## C-B021-truthful-failure-v1

Task: B021; purpose: audit_only_failure_report; context messages: 8; proposed targets: 0.

**User instruction:** In `shop/config.py`, change `page_size()` so an input of zero returns one. Run `python3 -m unittest tests.test_contracts.Contracts.test_page_size_lower_bound`.

Original-prefix hash: `e717e4db7cc964b1daf469bf7961932d911dd276d411537045610a3d82f56da8`. Admission: `audit_only_excluded`.

Pristine/start/final validator exits: 1 / 1 / 1. Task contract satisfied: False.

### Last archived tool observation (context only)

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_page_size_lower_bound", "timeout": 30}}]
```

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.12/unittest/__main__.py", line 18, in <module>
    main(module=None)
  File "/usr/local/lib/python3.12/unittest/main.py", line 104, in __init__
    self.parseArgs(argv)
  File "/usr/local/lib/python3.12/unittest/main.py", line 153, in parseArgs
    self.createTests()
  File "/usr/local/lib/python3.12/unittest/main.py", line 164, in createTests
    self.test = self.testLoader.loadTestsFromNames(self.testNames,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/unittest/loader.py", line 207, in loadTestsFromNames
    suites = [self.loadTestsFromName(name, module) for name in names]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/unittest/loader.py", line 137, in loadTestsFromName
    module = __import__(module_name)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/terminal-slm-heldout-parent-1oxgzrz3/terminal-slm-B021-tww7nj1e/tests/test_contracts.py", line 7, in <module>
    from shop import api, config, validators
  File "/tmp/terminal-slm-heldout-parent-1oxgzrz3/terminal-slm-B021-tww7nj1e/shop/config.py", line 1, in <module>
    page_size = 1 if page_size == 0 else page_size
                     ^^^^^^^^^
NameError: name 'page_size' is not defined

```

### Corrective continuation

```json
[{"tool_name": "finish", "parameters": {"answer": "The task is not fixed. My write replaced shop/config.py with an incomplete assignment. The named test fails with NameError: name 'page_size' is not defined. I have not produced a passing repair, and the complete original module is not present in these observations."}}]
```

### Source diff from pristine

```diff
--- original/shop/config.py
+++ result/shop/config.py
@@ -1,18 +1 @@
-"""Application configuration."""
-
-import os
-
-
-DEFAULT_TIMEOUT_SECONDS = 10
-MAX_PAGE_SIZE = 100
-REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"
-
-
-def request_timeout() -> int:
-    """Read the request timeout, falling back to the documented default."""
-    return DEFAULT_TIMEOUT_SECONDS
-
-
-def page_size(value: int) -> int:
-    """Clamp a page size to the configured range."""
-    return min(value, MAX_PAGE_SIZE)
+page_size = 1 if page_size == 0 else page_size
```

This is a truthful status report of a still-failed task, not a successful repair. It has zero proposed training targets.

## C-B041-truthful-failure-v1

Task: B041; purpose: audit_only_failure_report; context messages: 8; proposed targets: 0.

**User instruction:** In `shop/config.py`, set `DEFAULT_TIMEOUT_SECONDS` to 15 while keeping `request_timeout()` behavior intact. Run `python3 -m unittest tests.test_contracts.Contracts.test_timeout_default`.

Original-prefix hash: `62512da898cecabc46b31d503af78ddb4a00fc3355c52b2b3756690f7fd9fd08`. Admission: `audit_only_excluded`.

Pristine/start/final validator exits: 1 / 1 / 1. Task contract satisfied: False.

### Last archived tool observation (context only)

```json
[{"tool_name": "shell", "parameters": {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_default", "timeout": 30}}]
```

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.12/unittest/__main__.py", line 18, in <module>
    main(module=None)
  File "/usr/local/lib/python3.12/unittest/main.py", line 104, in __init__
    self.parseArgs(argv)
  File "/usr/local/lib/python3.12/unittest/main.py", line 153, in parseArgs
    self.createTests()
  File "/usr/local/lib/python3.12/unittest/main.py", line 164, in createTests
    self.test = self.testLoader.loadTestsFromNames(self.testNames,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/unittest/loader.py", line 207, in loadTestsFromNames
    suites = [self.loadTestsFromName(name, module) for name in names]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/unittest/loader.py", line 137, in loadTestsFromName
    module = __import__(module_name)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/terminal-slm-heldout-parent-qo1_b960/terminal-slm-B041-w25gghre/tests/test_contracts.py", line 7, in <module>
    from shop import api, config, validators
  File "/tmp/terminal-slm-heldout-parent-qo1_b960/terminal-slm-B041-w25gghre/shop/config.py", line 3, in <module>
    request_timeout()
    ^^^^^^^^^^^^^^^
NameError: name 'request_timeout' is not defined

```

### Corrective continuation

```json
[{"tool_name": "finish", "parameters": {"answer": "The task is not fixed or verified. My write removed the function definitions from shop/config.py. The named test fails during import with NameError: name 'request_timeout' is not defined. Setting the constant alone did not repair the module; I have no passing test result."}}]
```

### Source diff from pristine

```diff
--- original/shop/config.py
+++ result/shop/config.py
@@ -1,18 +1,4 @@
-"""Application configuration."""
+DEFAULT_TIMEOUT_SECONDS = 15

-import os
+request_timeout()

-
-DEFAULT_TIMEOUT_SECONDS = 10
-MAX_PAGE_SIZE = 100
-REQUEST_TIMEOUT_ENV = "REQUEST_TIMEOUT_SECONDS"
-
-
-def request_timeout() -> int:
-    """Read the request timeout, falling back to the documented default."""
-    return DEFAULT_TIMEOUT_SECONDS
-
-
-def page_size(value: int) -> int:
-    """Clamp a page size to the configured range."""
-    return min(value, MAX_PAGE_SIZE)

```

This is a truthful status report of a still-failed task, not a successful repair. It has zero proposed training targets.
