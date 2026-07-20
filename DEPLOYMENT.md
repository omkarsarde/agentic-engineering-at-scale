# Deploying the book as a website

Checked 2026-07-19. Recheck plan limits before choosing a host.

## Recommended: GitHub Pages

For a public repository, the expected hosting cost is **$0**. GitHub Pages is available for public repositories on GitHub Free, and this book already includes `.github/workflows/publish.yml`. The workflow validates the manuscript, runs every executable artifact test, renders the HTML-only site, and publishes `_book/` to the `gh-pages` branch.

1. Make `newbook/` the root of a public GitHub repository. If it must remain inside an outer repository, copy the workflow to the outer repository's `.github/workflows/`, add the following job-level default so every `run:` step executes inside the book, and add `path: newbook` under the Quarto publish action's `with:` block:

   ```yaml
   jobs:
     build-deploy:
       defaults:
         run:
           working-directory: newbook
       steps:
         # checkout, setup, install, validate, test, and render as supplied
         - name: Publish rendered HTML
           uses: quarto-dev/quarto-actions/publish@v2
           with:
             path: newbook
             target: gh-pages
             render: false
   ```
2. Push the source to `main`.
3. In repository settings, allow GitHub Actions read/write workflow permission if it is not already enabled.
4. Run the workflow manually once or push a commit.
5. Read the public URL under **Settings → Pages**. A normal project repository is usually auto-configured when the workflow creates `gh-pages`. If the repository itself is the special `<username>.github.io` user or organization site, manually set the Pages source to the root of the `gh-pages` branch after the first run.

The default `github.io` URL and TLS certificate cost nothing. A custom domain is optional; only the domain registration has an external cost. See the official [Quarto GitHub Pages guide](https://quarto.org/docs/publishing/github-pages.html) and [GitHub Pages availability](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

## Other $0 starting options

| Host | Current starting cost | Best fit | Important limit/status |
|---|---:|---|---|
| Cloudflare Pages | $0 | Git-connected static site, custom domain, CDN | Free plan currently lists 500 builds/month and 20,000 files/site. |
| Netlify Free | $0 | simple Git deploys and preview URLs | Free plan has a hard monthly credit limit; sites pause rather than silently charging when it is exhausted. |
| Quarto Pub | $0 | the simplest Quarto-native public experiment | Public only, 100 MB/site and soft 10 GB/month; Quarto says new projects should move to Posit Connect Cloud. |

Primary references: [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/), [Netlify pricing](https://www.netlify.com/pricing/), and [Quarto Pub status and limits](https://quarto.org/docs/publishing/quarto-pub.html).

GitHub Pages is the default here because it keeps source, review, tests, build, and deployment in one reproducible workflow. Cloudflare Pages is the strongest alternative when a private source repository or Cloudflare-managed domain is more convenient.

## Local rendering

Install Quarto, install `requirements.txt`, and run:

```powershell
python scripts/validate_book.py --strict
python -m pytest -q
quarto render --to html
```

The deployable static site appears in `_book/`. It contains ordinary HTML, CSS, JavaScript, SVG, search data, and assets, so any static host can serve it. There is no application server or database bill.

## Why the source is `.qmd`

Quarto Markdown remains readable in a text editor and in source control, while adding book navigation, cross-references, citations, equations, callouts, executable code blocks, multiple output formats, and Mermaid rendering. A plain `.md` file would work for prose, but `.qmd` gives the book-level publishing contract without forcing content into HTML.

The visual stack is intentionally layered:

- Mermaid for labeled architectures, lifecycles, state machines, and sequences;
- generated SVG for quantitative plots and comparisons;
- tables for exact mappings;
- runnable Python fixtures for behavior that a static picture cannot prove;
- ordinary raster images only when the content is inherently pictorial.

Every figure has a caption and text alternative. Diagrams are used to remove explanatory burden, not to decorate pages.
