# Publishing on GitHub

## Recommended: private while coursework is assessed

Create a new **private** repository named `MMA3001-Turbofan-RUL-Prediction`. Do not tick **Add README**, **Add .gitignore** or **Choose a license** (all files will be uploaded from this folder). Once assessment has finished, consider changing visibility to public after checking your course rules and reviewing the work.

## GitHub Desktop (no command line)

1. Extract the downloaded repository ZIP to a permanent folder.
2. Open GitHub Desktop → File → Add local repository → Choose folder. If prompted, choose **Create a repository here**.
3. Review the changes, write a commit summary such as `Initial FD001 project release`, then click **Commit to main**.
4. Click **Publish repository**; use the name above and keep **Private** selected.

## Git command line alternative

Run these inside the extracted project directory, after creating an **empty** GitHub repository:

```bash
git init
git add .
git commit -m "Initial FD001 turbofan RUL project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MMA3001-Turbofan-RUL-Prediction.git
git push -u origin main
```

GitHub will prompt for authentication through your configured credential manager, browser or token. Never place tokens or passwords in the repository.

## After publishing

Update the repository description to: *NASA C-MAPSS FD001 turbofan remaining useful life prediction using linear regression, random forest and neural networks.*

Optional topics: `machine-learning`, `predictive-maintenance`, `nasa-cmapss`, `turbofan`, `remaining-useful-life`, `python`, `scikit-learn`.

Add your fully revised report later only if appropriate for your course and sharing preferences.
