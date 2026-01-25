# Capstone Git & Sprint Workflow Guide

This document explains **how we use Git during sprints**, how to create branches, and how to collaborate without stepping on each other.

If you follow this, our repo stays clean and demos stay stress-free.

---

## 1. Core Principles

- **`main` is always runnable**
  - If it’s on `main`, it should work and be demoable.
- **Small, short-lived branches**
  - One feature or fix per branch.
  - Merge early and often.
- **No one commits directly to `main`**
  - Everything goes through a branch (and usually a PR).

---

## 2. Branching Strategy (Simple & Effective)

We use **trunk-based development**.

### Branches we use
- **`main`**
  - Stable, demo-ready branch
- **Feature branches**
  - Created from `main`
  - Named by purpose:
    - `feat/<feature-name>`
    - `fix/<bug-name>`
    - `chore/<cleanup-name>`

### Examples
feat/ring-buffer
feat/streamlit-dashboard
fix/reconnect-timeout
chore/project-structure
---
# 3. Sprint Workflow (Step-by-Step)

### 1️⃣ Start work on a feature
Always start from the latest `main`:

```bash
git checkout main
git pull
git checkout -b feat/your-feature-name
```
### 2️⃣ Work and commit locally
Make small, focused commits:

```bash
git status
git add <files>
git commit -m "feature: add ring buffer"
```

### 3️⃣ Push your branch
Push early (even if it's not finished):

```bash
git push -u origin feat/your-feature-name
```
This:
* backs up your work
* lets others see what you're doing
* enables early feedback

### 4️⃣ Open a Pull Request (PR)
When the feature:
* runs locally
* doesn't break the demo
Open a PR into main

### 5️⃣ Merge and delete branch
After review:

```bash
git checkout main
git pull
```
Delete your feature branch after merging.


Summary (Read This If Nothing Else)

* main = always demoable
* one feature per branch
* merge early, merge often
* tag sprint demos