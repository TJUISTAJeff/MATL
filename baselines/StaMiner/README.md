# StaMiner
 
Due to the copyright relationship with the enterprise, we are unable to disclose this part of data and its related code, so we public the core algorithm and Groum reproduction tool for reference.


## Introduction

```
├─EM_algorithm.py  # EM_algorithm core code.
├─GroumNode.py  # The GroumNode extraction tool for Python.
```

## Groum Node
It's a python version of the paper <a  href ="https://dl.acm.org/doi/10.1145/1595696.1595767">Groum</a> reproduction.
In GroumNode.py, line 637 and 638 need to be modified as the the file need to be parsed and the output dir directively.

## EM_algorithm
The line 44 and 45 represent the corresponding sentences of two language, which is both a list of string lists.

For example:
`[['This', 'is', 'Stainer']]` and `[['这', '是', 'StaMiner']]`
