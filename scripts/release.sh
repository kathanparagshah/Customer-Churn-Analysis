#!/bin/bash

# Release script for Bank Churn Analysis package
# Usage: ./scripts/release.sh [version] [type]
# Example: ./scripts/release.sh 0.2.1 test
# Example: ./scripts/release.sh 0.2.1 prod

set -e

VERSION=${1:-"0.2.1"}
TYPE=${2:-"test"}

echo "🚀 Starting release process for version $VERSION (type: $TYPE)"

# Validate inputs
if [[ ! "$TYPE" =~ ^(test|prod)$ ]]; then
    echo "❌ Error: Type must be 'test' or 'prod'"
    exit 1
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "⚠️  Warning: You're not on the main branch (current: $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo "❌ Error: You have uncommitted changes. Please commit or stash them first."
    git status --short
    exit 1
fi

# Update version in files
echo "📝 Updating version to $VERSION..."

# Update pyproject.toml
sed -i.bak "s/version = \"[^\"]*\"/version = \"$VERSION\"/" pyproject.toml

# Update setup.py
sed -i.bak "s/version=\"[^\"]*\"/version=\"$VERSION\"/" setup.py

# Remove backup files
rm -f pyproject.toml.bak setup.py.bak

# Commit version changes
echo "💾 Committing version changes..."
git add pyproject.toml setup.py
git commit -m "Bump version to $VERSION"

# Create and push tag
if [[ "$TYPE" == "prod" ]]; then
    TAG="v$VERSION"
    echo "🏷️  Creating production tag: $TAG"
else
    TAG="$VERSION-test"
    echo "🏷️  Creating test tag: $TAG"
fi

git tag "$TAG"

echo "📤 Pushing changes and tag..."
git push origin main
git push origin "$TAG"

echo "✅ Release process completed!"
echo "📊 Monitor the GitHub Actions workflow at:"
echo "   https://github.com/kathanparagshah/Customer-Churn-Analysis/actions"

if [[ "$TYPE" == "test" ]]; then
    echo "🧪 Test release will be published to TestPyPI"
    echo "📦 Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bank-churn-analysis==$VERSION"
else
    echo "🎉 Production release will be published to PyPI"
    echo "📦 Install with: pip install bank-churn-analysis==$VERSION"
fi

echo "🔗 GitHub Release: https://github.com/kathanparagshah/Customer-Churn-Analysis/releases/tag/$TAG"