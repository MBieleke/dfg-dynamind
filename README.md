# How to contribute to this repository

This guide walks you through contributing using Visual Studio Code (VS Code). Each step shows both a **click-based** and a **terminal-based** approach — use whichever feels more natural to you.

---

## One-time setup

### 1. Install the tools
- Download and install [Git](https://git-scm.com/downloads)
- Download and install [Visual Studio Code](https://code.visualstudio.com/)

### 2. Install VS Code extensions
Click the **Extensions icon** in the left sidebar (four squares) and install:
- **GitHub Pull Requests** (by GitHub)
- **Git Graph** (by mhutchie)

### 3. Sign into GitHub in VS Code
- Click the **Accounts icon** at the bottom of the left sidebar
- Click **Sign in with GitHub** and follow the browser prompts

### 4. Clone the repository
This downloads the project to your computer.

**By clicking:**
- Press `Ctrl+Shift+P` → type `Git: Clone` → paste the URL below → choose a folder

**In the terminal** (**Terminal → New Terminal**):
```bash
git clone https://github.com/MBieleke/dfg-dynamind.git
cd dfg-dynamind
```

### 5. Set your identity
Run these once in the terminal — there is no click alternative for this:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@university.de"
```

---

## Every time you want to make a change

### Step 1 — Sync with the latest version
Always do this before starting any work.

**By clicking:**
Source Control icon (left sidebar) → **⋯** → **Pull**

**In the terminal:**
```bash
git checkout main
git pull
```

### Step 2 — Create a branch for your change
A branch is your own isolated workspace. Nothing you do here affects the main project until your supervisor approves it.

Name your branch something descriptive:
- `experiment/add-practice-trials`
- `fix/timing-issue-block2`
- `experiment/new-feedback-screen`

**By clicking:**
Click the **branch name in the bottom-left corner** (says `main`) → **Create new branch** → type the name → press Enter

**In the terminal:**
```bash
git checkout -b experiment/your-branch-name
```

The branch name in the bottom-left corner of VS Code will update to confirm you are on your new branch.

### Step 3 — Make your changes
Edit files in VS Code as normal and run the experiment to verify everything works.

### Step 4 — Save your changes to Git (commit)
A commit is a saved snapshot of your work. Write a short message describing what you changed and why.

**By clicking:**
- Source Control icon → click **+** next to each changed file to stage it
- Type a message in the **Message** box (e.g. `fix audio feedback on block 3`)
- Click **✓ Commit**

**In the terminal:**
```bash
git add .
git commit -m "fix audio feedback on block 3"
```

Good commit messages:
- `Add fixation cross to practice trials`
- `Fix audio feedback not playing on block 3`
- `Increase SOA from 500ms to 750ms in experiment 2`

Commit as often as you like — after each meaningful change is better than one big commit at the end.

### Step 5 — Upload your branch to GitHub
**By clicking:**
Source Control icon → **⋯** → **Push** (click **Publish Branch** if prompted)

**In the terminal:**
```bash
git push origin your-branch-name
```

### Step 6 — Open a Pull Request
A Pull Request is how you ask your supervisor to review and merge your changes.

**By clicking:**
- Click the **GitHub icon** in the left sidebar (cat logo)
- Under **Pull Requests** → click **+**
- Set base branch to `main`, compare branch to your branch
- Fill in the description template
- Click **Create**

**Via github.com:**
- Go to the repository on GitHub
- Click the **Compare & pull request** banner that appears
- Fill in the template and click **Create pull request**

Your supervisor will be notified automatically.

### Step 7 — Responding to review comments
If your supervisor requests changes, make the adjustments in your files, then repeat Steps 4 and 5. The Pull Request updates automatically — no need to open a new one.

Once approved, your supervisor will merge your changes and your branch will be deleted automatically.

---

## Important rules

- **Never push directly to `main`** — always work on a branch and open a Pull Request
- **Always pull before starting new work** — this avoids conflicts with others' changes
- **One branch per change** — don't mix unrelated changes in the same branch
- **Never commit participant data** — raw data files should stay on your local machine only

---

## Quick reference

| Situation | Click | Terminal |
|---|---|---|
| Sync latest version | Source Control → ⋯ → Pull | `git pull` |
| Create a branch | Branch name (bottom-left) → Create new branch | `git checkout -b name` |
| Stage changes | Source Control → **+** next to files | `git add .` |
| Commit | Source Control → message → ✓ Commit | `git commit -m "message"` |
| Push to GitHub | Source Control → ⋯ → Push | `git push origin branch-name` |
| Open a Pull Request | GitHub icon → Pull Requests → + | — (use GitHub.com) |
| View history | Git Graph icon in left sidebar | `git log --oneline` |

---

## Something went wrong?

Don't panic — Git is designed so that mistakes are hard to make permanent. Before trying to fix anything yourself, contact your supervisor and describe what happened. In most cases everything can be recovered.