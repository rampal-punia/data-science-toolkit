# Essential Git Commands for Data Science

## Configuration

```bash
# Set your name for all local repositories
git config --global user.name "Your Name"

# Set your email for all local repositories
git config --global user.email "your.email@example.com"
```

## Repository Setup

```bash
# Initialize a new Git repository in the current directory
git init

# Clone an existing repository
git clone https://github.com/user/repository.git [directory_name]
# If [directory_name] is omitted, Git will create a directory based on the repository name
```

## Basic Commands

```bash
# Check the status of your working directory
git status

# View commit history
git log
# For a more concise, one-line format:
git log --oneline

# Add a specific file to the staging area
git add filename

# Add all changed files to the staging area
git add .

# Commit staged changes
git commit -m "Descriptive commit message"
```

## Branching and Merging

```bash
# List all branches (* indicates the current branch)
git branch

# Create a new branch
git branch branch_name

# Switch to a different branch
git checkout branch_name

# Create and switch to a new branch in one command
git checkout -b new_branch_name

# Merge a branch into the current branch
git merge branch_name

# Delete a branch (use -D instead of -d to force deletion)
git branch -d branch_name
```

## Remote Repositories

```bash
# Add a remote repository
git remote add origin https://github.com/user/repository.git

# Fetch changes from a remote repository and merge into current branch
git pull origin branch_name

# Push local changes to a remote repository
git push origin branch_name

# Push a new local branch to a remote repository
git push -u origin new_branch_name
```

## Stashing

```bash
# Temporarily store modified, tracked files
git stash

# List all stashes
git stash list

# Apply a stash and remove it from the stash list
git stash pop

# Apply a specific stash
git stash apply stash@{index}
```

## Advanced Operations

```bash
# Rebase current branch onto another branch
git rebase branch_name

# Interactive rebase for editing, squashing, or reordering commits
git rebase -i HEAD~n  # n is the number of commits to go back

# Cherry-pick a specific commit to apply to the current branch
git cherry-pick commit_hash

# Revert a commit, creating a new commit with the inverse of the reverted commit
git revert commit_hash

# Soft reset: move HEAD to a previous commit, but keep changes staged
git reset --soft commit_hash

# Hard reset: move HEAD to a previous commit and discard all changes
git reset --hard commit_hash

# Amend the most recent commit
git commit --amend -m "New commit message"
```

## Tagging

```bash
# Create an annotated tag
git tag -a v1.0 -m "Version 1.0"

# List all tags
git tag

# Push tags to a remote repository
git push origin --tags
```

## Inspection and Comparison

```bash
# Show the author and revision of each line in a file
git blame filename

# Show differences between two commits
git diff commit_hash1 commit_hash2

# Show differences between working directory and last commit
git diff
```

## Working with .gitignore

```bash
# Remove a cached directory (useful when adding to .gitignore)
git rm -r --cached directory_name
git commit -m "Stop tracking directory_name"

# Add .gitignore file to version control
git add .gitignore
git commit -m "Add .gitignore file"
```

## Submodules

```bash
# Add a submodule to your repository
git submodule add https://github.com/user/repository.git

# Initialize submodules in a cloned repository
git submodule init

# Update submodules to their latest commits
git submodule update --remote
```

Remember to replace `commit_hash`, `branch_name`, `filename`, etc., with actual values when using these commands. Always be cautious when using commands that can potentially discard changes (like `git reset --hard`).

---

# Git Workflow Guide: Managing Master, Development & Feature Branches

## Setup

1. Ensure you have both `master` and `development` branches locally and on GitHub.
2. Set `development` as your default working branch:
   ```
   git checkout development
   ```

## Daily Workflow

1. Start your workday by updating your local `development` branch:
   ```
   git checkout development
   git pull origin development
   ```

2. Create a new feature branch for your work:
   ```
   git checkout -b feature/your-feature-name
   ```

3. Work on your feature, making commits as you go:
   ```
   git add .
   git commit -m "Descriptive commit message"
   ```

4. When your feature is complete, update your local `development` branch and merge it into your feature branch:
   ```
   git checkout development
   git pull origin development
   git checkout feature/your-feature-name
   git merge development
   ```

5. Resolve any merge conflicts if they occur.

6. Push your feature branch to GitHub:
   ```
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub from your feature branch to the `development` branch.

8. After review and approval, merge the pull request on GitHub.

9. Delete the feature branch on GitHub after merging.

10. Update your local `development` branch and delete the local feature branch:
    ```
    git checkout development
    git pull origin development
    git branch -d feature/your-feature-name
    ```

## Updating Master

Periodically (e.g., for releases), you'll want to update the `master` branch:

1. Ensure `development` is up to date:
   ```
   git checkout development
   git pull origin development
   ```

2. Merge `development` into `master`:
   ```
   git checkout master
   git merge development
   ```

3. Push the updated `master` to GitHub:
   ```
   git push origin master
   ```

## Merging Between Master and Development

When merging between `master` and `development`, the order matters:

1. Merging `development` into `master` (Recommended for releases):
   ```
   git checkout master
   git merge development
   ```
   - This brings new features and changes from `development` into `master`.
   - Typically done for releases or major updates.

2. Merging `master` into `development` (Less common):
   ```
   git checkout development
   git merge master
   ```
   - This brings changes from `master` into `development`.
   - Usually done to sync `development` with hotfixes made directly to `master`.

Key considerations:
- The branch you're on (checked out) is the one receiving changes.
- Merging `development` into `master` moves new features to production.
- Merging `master` into `development` syncs critical fixes back to the development branch.
- Resolve conflicts on the branch you're merging into.

## Tips

- Always check which branch you're on with `git branch` before starting work.
- Use `git status` frequently to see your current state.
- Consider using a Git GUI tool like GitKraken or SourceTree for a visual representation of your branches.
- Set up Git aliases for common commands to save time.

## Branch Cleanup

Periodically, clean up old feature branches:

1. Update your local branch list:
   ```
   git fetch -p
   ```

2. List merged branches:
   ```
   git branch --merged
   ```

3. Delete old feature branches:
   ```
   git branch -d feature/old-feature-name
   ```

## Workflow for Contributors

Contributors should follow these guidelines:

1. Fork the repository:
   - On GitHub, create a personal fork of the main project repository.

2. Clone the fork locally:
   ```
   git clone https://github.com/your-username/project-name.git
   cd project-name
   ```

3. Add the original repository as an upstream remote:
   ```
   git remote add upstream https://github.com/original-owner/project-name.git
   ```

4. Create a feature branch off of `development`:
   ```
   git checkout development
   git pull upstream development
   git checkout -b feature/your-feature-name
   ```

5. Work on the feature, committing changes to your feature branch.

6. Regularly sync your `development` branch with the upstream:
   ```
   git checkout development
   git pull upstream development
   ```

7. Rebase your feature branch on the updated `development`:
   ```
   git checkout feature/your-feature-name
   git rebase development
   ```

8. Push your feature branch to your fork:
   ```
   git push origin feature/your-feature-name
   ```

9. Create a pull request from your feature branch to the `development` branch of the main repository.

10. After review and approval, a project maintainer will merge the pull request.

11. After merging, sync your fork:
    ```
    git checkout development
    git pull upstream development
    git push origin development
    ```

12. Delete your local feature branch:
    ```
    git branch -d feature/your-feature-name
    ```

Key points for contributors:
- Always work on feature branches, never directly on `master` or `development`.
- Keep your `development` branch updated with the upstream repository.
- Use pull requests for all contributions, even if you have write access to the repository.
- Communicate with maintainers about larger changes before investing significant time.

Remember, communication with your team about branch usage and merging strategies is key to a smooth workflow.