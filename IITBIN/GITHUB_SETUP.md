# GitHub Setup Guide

## Prerequisites

1. **Install Git** (if not already installed):
   - Download from: https://git-scm.com/download/win
   - Or install via package manager (Chocolatey, Scoop, etc.)

2. **Create GitHub Account** (if you don't have one):
   - Go to: https://github.com
   - Sign up for a free account

## Step-by-Step Instructions

### 1. Install Git (if needed)

**Option A: Download from official website**
- Go to https://git-scm.com/download/win
- Download and run the installer
- Follow the installation wizard

**Option B: Using Chocolatey (if you have it)**
```powershell
choco install git
```

**Option C: Using Scoop (if you have it)**
```powershell
scoop install git
```

### 2. Configure Git (first time setup)

Open PowerShell/Command Prompt and run:

```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### 3. Create GitHub Repository

1. Go to https://github.com
2. Click the "+" icon in the top right
3. Select "New repository"
4. Repository name: `cross-modal-knowledge-transfer` (or your preferred name)
5. Description: `Cross-Modal Knowledge Transfer using EEG, Eye-tracking, and GSR data`
6. Make it **Public** or **Private** (your choice)
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

### 4. Initialize and Push to GitHub

Open PowerShell in your project directory (`C:\Users\Harsh\IITBIN`) and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Cross-Modal Knowledge Transfer application

- Multi-modal data processing for EEG, eye-tracking, and GSR
- Domain adaptation and modality dropout
- Contrastive learning and cross-modal attention
- Comprehensive evaluation framework
- Training pipeline with configuration management
- Complete documentation and usage examples"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/cross-modal-knowledge-transfer.git

# Push to GitHub
git push -u origin main
```

### 5. Alternative: Using GitHub CLI (if you have it)

If you have GitHub CLI installed:

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: Cross-Modal Knowledge Transfer application"

# Create repository on GitHub and push
gh repo create cross-modal-knowledge-transfer --public --source=. --remote=origin --push
```

## Repository Structure

Your GitHub repository will contain:

```
cross-modal-knowledge-transfer/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model architectures
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Evaluation framework
│   └── utils/             # Utilities
├── configs/               # Configuration files
├── docs/                  # Documentation
├── results/               # Training results (gitignored)
├── data/                  # Dataset storage (gitignored)
├── requirements.txt       # Dependencies
├── train.py              # Main training script
├── README.md             # Project description
├── QUICK_START.md        # Quick start guide
└── .gitignore            # Git ignore rules
```

## After Pushing

1. **Add a README**: Your repository will show the README.md content
2. **Add Topics**: Go to repository settings and add topics like:
   - `machine-learning`
   - `cross-modal`
   - `eeg`
   - `eye-tracking`
   - `domain-adaptation`
   - `pytorch`
3. **Create Releases**: Tag important versions
4. **Enable Issues**: For bug reports and feature requests

## Future Updates

To update your repository:

```bash
# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Troubleshooting

### Authentication Issues
If you get authentication errors:

1. **Use Personal Access Token**:
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate new token with repo permissions
   - Use token as password when prompted

2. **Use SSH** (recommended):
   ```bash
   # Generate SSH key
   ssh-keygen -t ed25519 -C "your.email@example.com"
   
   # Add to GitHub (copy public key to GitHub Settings > SSH keys)
   cat ~/.ssh/id_ed25519.pub
   
   # Change remote URL to SSH
   git remote set-url origin git@github.com:YOUR_USERNAME/cross-modal-knowledge-transfer.git
   ```

### Large Files
If you have large data files:
- Use Git LFS: `git lfs track "*.npy"`
- Or add them to .gitignore

## Repository Features

Once pushed, your repository will have:

✅ **Professional README** with project description  
✅ **Complete documentation** in docs/ folder  
✅ **Working code** ready to run  
✅ **Configuration examples** for different use cases  
✅ **Proper .gitignore** to exclude unnecessary files  
✅ **Clear project structure** for easy navigation  

## Next Steps

1. **Share the repository** with collaborators
2. **Create issues** for bugs or feature requests
3. **Add collaborators** if working in a team
4. **Set up CI/CD** for automated testing (optional)
5. **Create releases** for stable versions

Your Cross-Modal Knowledge Transfer application is now ready to be shared with the world! 🌟
