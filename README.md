# 🛠 Bank Statement Backend - Git Workflow Guide for Interns

Welcome! This document explains the Git workflow and rules for contributing to this repository. Please **read carefully** and follow each step while working on your assigned tasks.

---

## Branch Structure

| Branch                        | Purpose                                              |
|-------------------------------|------------------------------------------------------|
| `main`                        | Production-ready code. **Protected.**                |
| `feature/*`                   | feature-branch                                       |

---

## 🔐 Branch Protection Rules

- 🔒 `master`: **No direct pushes.** All updates via approved pull requests.
- ✅ You can create and push `feature/*` branches freely under your respective team branches.

---
## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kreativetimebox/bank-statement-categorization-model.git
cd bank-statement-categorization-model
```

### 2. Create and switch to a new feature branch
Always base your work on the latest `main` branch.

```bash
git checkout main
git pull origin main

# Create a feature branch:
git checkout -b feature/yourname-taskname
```

---

## Working on Your Task

1. Make your code changes
2. Add and commit with a clear message:

```bash
git add .
git commit -m "Add: [Short Description of Your Feature]"
```

3. Push to your branch:

```bash
git push origin feature/yourname-taskname
```

---

## Create a Pull Request (PR)

Go to the repository on GitHub and:

- Click **“Compare & Pull Request”**
- Base branch: `main`
- Head branch: `feature/yourname-taskname`
- Add a meaningful PR title and description
- Click **“Create Pull Request”**

---

## ✅ After Review

- Your PR will be reviewed and merged into `main`

---

## 🧪 Testing Setup (Do Not Skip)

Before you submit your PR:

- ✅ Pull the latest `main` changes:  
  ```bash
  git pull origin main --rebase
  ```
- ✅ Test your code locally
- ✅ Ensure your changes do not break existing features

---

## ❌ Don’t Do

- 🚫 Do **not** push to `main`
- 🚫 Do **not** work on `main` directly
- 🚫 Do **not** merge your own PRs unless told

---

## 👥 Need Help?

Contact your project lead if:
- You can't push your branch
- You’re unsure about the workflow
- You need feedback on your PR

Happy coding!

