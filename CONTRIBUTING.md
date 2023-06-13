# Contributing

## Code of Conduct

Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before contributing.

## Getting started

### How to submit a Contribution

1. Create your own fork of the code
2. Do the changes in your fork
3. If you like the change and think the project could use it:
    * Be sure you have followed the code style for the project.
    * Send a pull request.

### Set-up

Expects:
Env manager: conda
Os: Mac
Package manager: pip
pre-commit

1. Setup a new conda env: `conda create -n r5py python=3.9.13`
2. Activate the environment: `conda activate r5py`
3. Launch terminal and change directory to wherever you keep your GitHub repos: `cd ~/Documents`
4. Clone this repo, eg with https: `git clone https://github.com/datasciencecampus/transport-network-performance.git`
5. Change directory to the repo: `cd transport-network-performance`
6. Install pre-commit hooks: `pre-commit install`
7. Update pip: `pip install --upgrade pip`
8. Install r5py & other reqs: `pip install -r requirements.txt`

**Java Development Kit**
Openjdk 11 is recommended by r5py docs. [sdkman](https://sdkman.io/) is used here.

9. Get sdkman: `curl -s "https://get.sdkman.io" | bash`
10. Follow instructions in terminal, you’ll be asked to: `source “<SOME_PATH>.sdkman/bin/sdkman-init.sh"`
This will add sdkman env variables to: `~/.bash_profile` and possibly `~/.zshrc` too.
11. To see versions available on your os: `sdk list java`
12. Install the required version using the `identifier` column, eg: `sdk install java 11.0.19-amzn`
13. Check this is the currenty used java version: `sdk current java`

**Data**

14. Get some pdf data from geofabrik download server, eg: [Geofabrik Wales latest](https://download.geofabrik.de/europe/great-britain/wales.html).
15. Get some GTFS data, eg from [DfT BODS](https://data.bus-data.dft.gov.uk/), **account required**.
16. Store in `data/external`. These files should be gitignored but please check.
17. Now you should be able to run the check setup script, from terminal: `python3 src/utils/check-setup.py`
18. If everything is working as expected, you should see some Java flavoured warnings about `--illegal-access` that you can ignore. But importantly look out for the message: `r5py has created the expected database files.`
19. If you've made it this far, you've earned yourself a coffee.


### Pre-commit

*Placeholder section*

### Note on Issues, Feature Requests, and Pull Requests

The Campus looks at issues, features requests, and pull requests on a regular basis but cannot unfortunately guarantee prompt implementation or response.

### How to report a bug/issue

If you find a security vulnerability, do NOT open an issue. Email datacampus@ons.gov.uk instead.

When filing an issue, make sure to answer the questions in the Bug template.

### How to request a new feature

When raising an issue, select the 'Feature request' option and answer the questions in the template.

## Code conventions

We mainly follow the [Quality Assurance of Code for Analysis and Research](https://best-practice-and-impact.github.io/qa-of-code-guidance/intro.html) in our code conventions.

## Testing

*Placeholder section*

## Code coverage

*Placeholder section*

## Documentation

*Placeholder section*
