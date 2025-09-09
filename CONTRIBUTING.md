# Contributing to Mainrun

We welcome contributions to Mainrun! Whether you've found a bug, have a feature request, or want to contribute code, we appreciate your input.

## How to Contribute

### Reporting Issues

Found a bug or have a suggestion? Please [open an issue](https://github.com/maincodehq/mainrun/issues) with:

- **Bug Reports**: Include steps to reproduce, expected vs actual behavior, and environment details
- **Feature Requests**: Describe the enhancement and why it would be valuable
- **Questions**: Ask about unclear documentation or implementation details

### Submitting Code Changes

We love pull requests! Here's how to contribute code:

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/mainrun.git
   cd mainrun
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-improvement
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed
   - Ensure `task train` still works

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Clear description of your changes"
   git push origin feature/your-improvement
   ```

5. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Describe your changes and why they're valuable

## What We're Looking For

### Good First Issues

Look for issues labeled `good first issue` - these are great starting points for new contributors.

### Areas of Interest

- Performance optimizations
- Documentation improvements
- Bug fixes
- New features that enhance the assessment experience
- Better error messages and debugging tools
- Cross-platform compatibility improvements

## Code Standards

- **Python**: Follow PEP 8, use type hints where appropriate
- **JavaScript**: Use modern ES6+ syntax
- **Documentation**: Clear, concise, and helpful
- **Commits**: Descriptive messages explaining what and why

## Testing

Before submitting:
1. Run `task train` to ensure the training pipeline works
2. Test any new commands or features
3. Verify documentation is accurate

## Questions?

- Open an issue for discussion
- Tag maintainers for complex changes
- Be patient - we review PRs as time allows

## Recognition

Contributors will be acknowledged in our release notes. Significant contributions may lead to collaboration opportunities.

Thank you for helping make Mainrun better for everyone!