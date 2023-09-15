# `docs` folder overview

This folder contains all the source files needed to build package documentation
using [`sphinx`](https://www.sphinx-doc.org/en/master/).

## Building the documentation locally

This is useful whilst editing the documentation locally and previewing new
additions/edits. Following the steps below will render the documenation locally
allowing you to check for any warnings or errors during the build stage.

1. Ensure the dependencies in `requirements.txt` have been installed. This will
install `sphinx`, the necessary themes, and all the other Python dependecies
for this package.

2. Call the following from the project root:

    ```bash
    make -C docs/ html
    ```

    Or, from the within this docs directory:

    ```bash
    make html
    ```

    > Note: On Windows, if you are using PowerShell the make command may not
    work. If this is the case, you should be able to run `.\make.bat html`
    after navigating to this directory.

    Calling one of the commands above will trigger `sphinx-build` and render
    the documentaion in HTML format within the `build` directory.

3. Inside `docs/build/html/`, opening/refreshing `index.html` in a browser will
display the documentation landing page.

## Cleaning the docs folder

From time to time, it maybe necessary to clean the build folder (e.g., to
unpick some edits that have not made their way through to the browser for some
reason).

> Note: `sphinx-build` will only rebuild pages if the respective source file(s)
has changed. Calling clean maybe helpful to either force an entire rebuild of
all pages, or include an update that isn't picked up via a source (e.g. a CSS
file update).

To clean the build folder, call the following:

```bash
# from the project root
make -C docs/ clean

# or, from within the docs folder
make clean
```

It's also possible to combine both the clean and HTML build commands together
as follows:

```bash
# from the project root
make -C docs/ clean html

# or, from within the docs folder
make clean html
```

> Note: the contents of the `docs/build` folder are ignored by Git. Cleaning
the build folder locally will therefore only impact your local documentation
build.

## Building the documentation 'on push' to a remote branch

There is a GitHub action set-up (`.github/workflows/sphinx-render.yml`) that
runs on all pushes to any branch. This will attempt to build the `docs/source`
folder content and will fail if `sphinx-build` throws any errors or warnings.
This helps ensure the quality of the documentation on each push and allows
developers to correct any issues sooner.

The deployment stage of this GitHub action is only done when pushing to the
`dev` branch (i.e. after merging in a PR). Therefore, any changes made to
`docs` in a feature branch will not appear in the deployed documentation.

> Note: the current implementation of the GitHub action deploys on push to
`dev` but this is subject to change at a later date. It will likely be change
to puses to `main` once an inital release of this package is available.
