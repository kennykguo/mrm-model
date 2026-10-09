# Independent implementation

On 2026-10-09 the user requested a fresh implementation, without using the
existing `mrm-model` code or any of its branches. The `v1-3` worktree was recreated
as an empty orphan branch. Before new work began, `git status` reported
`No commits yet on v1-3`, and the directory contained only its `.git` worktree link.

Earlier audit work was moved outside this branch and is not a source for this
implementation. No source files, extracted data, tests, design documents, or
Python environment were imported from it. New research agents began without
the audit conversation. The permitted starting references are the user's
attachment, the supplied OFC 2026 paper, the interactive textbook, the photonics
simulations repository, and independently consulted primary publications.

This document records the development boundary. It is not a claim that familiar
physical equations or general software techniques are novel.
