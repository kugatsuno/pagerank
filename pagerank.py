import os
import re
import sys
import random
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    output_dict = {}
    linked_pages = corpus[page]
    total_pages = len(corpus)

    if linked_pages is None:
        equal_probability = 1 / total_pages
        for key in corpus:
            output_dict[key] = equal_probability
        return output_dict
    else:
        equal_probability = (1 - damping_factor) / total_pages
        linked_pages_probability = damping_factor / len(linked_pages)
        for key in corpus:
            if key in linked_pages:
                output_dict[key] = equal_probability + linked_pages_probability
            else:
                output_dict[key] = equal_probability
        return output_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result_dict = {}
    output_dict = {}
    corpus_list = []
    for key in corpus:
        corpus_list.append((key))
    first_page = random.choice(corpus_list)
    generated_page = '1.html'

    for i in range(n):
        if i == 0:
            transition_model_dict = transition_model(corpus, first_page, damping_factor)
        else:
            transition_model_dict = transition_model(corpus, generated_page, damping_factor)
        keys, values = zip(*transition_model_dict.items())
        generated_page = np.random.choice(a=keys, p=values)
        if generated_page in result_dict:
            result_dict[generated_page] = result_dict[generated_page] + 1
        else:
            result_dict[generated_page] = 1

    for key in result_dict:
        output_dict[key] = result_dict[key] / n

    return output_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_dict = {}
    return_dict = {}
    total_pages = len(corpus)

    for key in corpus:
        current_dict[key] = 1/total_pages

    while True:
        for key in corpus:
            links_to_key = {}
            for check in corpus:
                if key in corpus[check]:
                    links_to_key[check] = len(corpus[check])
                if len(corpus[check]) == 0:
                    links_to_key[check] = total_pages
            if links_to_key:
                linked_probability = 0
                for link in links_to_key:
                    linked_probability = linked_probability + (current_dict[link]/links_to_key[link])
                return_dict[key] = ((1-damping_factor)/total_pages) + damping_factor*linked_probability
            else:
                return_dict[key] = ((1-damping_factor)/total_pages)

        complete_check = True
        for key_check in return_dict:
            if abs(return_dict[key_check] - current_dict[key_check]) > 0.001:
                complete_check = False
                current_dict[key_check] = return_dict[key_check]

        if complete_check:
            return return_dict


if __name__ == "__main__":
    main()
