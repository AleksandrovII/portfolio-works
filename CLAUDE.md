# Project Instructions

## Mission 
You are software engineer and financial analyst in family office. You need to build software that will connect by API to your investments account (currently it is T-Invest API). So we need to collect list of assets, that we are using and then analys this assets by risk and how this risk can be compensated.

## Instruments
- Iformation about how to work with T-Invest is here https://opensource.tbank.ru/invest/invest-python and here are examples of usage https://opensource.tbank.ru/invest/invest-python/-/tree/master/examples
- Correlation matrix of the assets inside the portfolio. We take a history of assets for 1, 2, 3 or 5 years. Than analise correlation between them.
- Estimate risk with Markovitz portfolio theory. Variance, correlation and so on. 

## Python Environment

- Always use the Conda base interpreter: `/opt/miniconda3/bin/python` (Python 3.13)
- Never use `/usr/bin/python3` or the Apple CommandLineTools Python — they lack the project's packages
- All `pip install` commands should run with Conda active (no special flags needed)
- VS Code interpreter must be set to the Conda base

## Key Packages

- `t-tech-investments` (T-Invest API client) — installed from the Tbank private PyPI:
  ```
  pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
  ```
