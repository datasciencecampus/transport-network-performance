# `docs` folder overview

This folder contains all the source files needed to build package documentation.

## Using Quarto with Quartodoc

To build the documentation using Quarto and quartodoc, you will need to install [Quarto](https://quarto.org/docs/get-started/).

The Python package, `quartodoc`, is included in the project 'requirements.txt'.

Run `quartodoc build` in the top-level folder; this will create a new directory, `docs/reference/` and populate it with automatically generated `.qmd` files for each reference documentation page.

Then run `quarto preview` to view a local rendering of the website.

Adding to the docs can be done as follows:

- New tutorials can be added to `docs/tutorials/`.
- New How-to guides can be added to `docs/how_to/`.
- New explanation guides can be added to `docs/explanation/`.
- In all cases, create a new subdirectory in one of the above areas and then create an `index.qmd` file to add page content. All assets pertaining to the new page can then also be stored within this sub directory.


## Building the documentation 'on push' to a remote branch

There is a GitHub action set-up (`.github/workflows/quarto-render.yml`) that
runs on all pushes to the `dev` and `main` branches. This will attempt to render
the content within `docs/` and then deploy them to the `gh-pages` branch.
