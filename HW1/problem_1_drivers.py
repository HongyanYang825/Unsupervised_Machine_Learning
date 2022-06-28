'''
    DS 5230
    Summer 2022
    HW1_Problem_1_Aminer_basic_dataset_analysis_drivers

    Drivers to support basic dataset analysis to the Aminer dataset

    Hongyan Yang
'''


import numpy as np
import matplotlib.pyplot as plt

def parse_file(path):
    i, lines_list = 0, []
    with open(path, encoding = "utf-8") as f:
        while i < 2002:
            line = f.readline()
            lines_list.append(line)
            i += 1
    return lines_list

def write_file(lines_list, out_path):
    with open(out_path, "a", encoding = "utf-8") as f:
        for each in lines_list:
            f.write(each)

def extract_attributes(path):
    authors_set, venues_set = set(), set()
    publications_set, citations_set = set(), set()
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#@"):
                line = line.lstrip("#@").rstrip("\n")
                temp_list = line.split(", ")
                temp_list = [each.strip() for each in temp_list]
                authors_set.update(temp_list)
            elif line.startswith("#c"):
                line = line.lstrip("#c").rstrip("\n")
                venues_set.add(line.strip())
            elif line.startswith("#index"):
                line = line.lstrip("#index").rstrip("\n")
                publications_set.add(line.strip())
            elif line.startswith("#%"):
                line = line.lstrip("#%").rstrip("\n")
                citations_set.add(line.strip())
    print(f"There are {len(authors_set)} distinct authors, {len(venues_set)} distinct"\
          f" publication venues, {len(publications_set)} distinct publications, "\
          f"and {len(citations_set)} distinct citations/references.")
    return (len(authors_set),len(venues_set),len(publications_set),
            len(citations_set))

def look_up_venue(path):
    related_venues = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
           if line.startswith("#c"):
               if ("Principles and Practice of Knowledge Discovery in Databases"
                   in line):
                   line = line.lstrip("#c").rstrip("\n")
                   related_venues.append(line.strip())
    return related_venues, set(related_venues)

def author_publications(path):
    author_publications_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#*"):
                line = line.lstrip("#*").rstrip("\n")
                publication = line.strip()
            if line.startswith("#@"):
                line = line.lstrip("#@").rstrip("\n")
                temp_list = line.split(", ")
                temp_list = [each.strip() for each in temp_list]
                for each in temp_list:
                    if each in author_publications_dict:
                        author_publications_dict[each].append(publication)
                    else:
                        author_publications_dict[each] = [publication]
    return author_publications_dict    

def plot_author_publications(path):
    author_publications_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#@"):
                line = line.lstrip("#@").rstrip("\n")
                temp_list = line.split(", ")
                temp_list = [each.strip() for each in temp_list]
                for each in temp_list:
                    if each in author_publications_dict:
                        author_publications_dict[each] += 1
                    else:
                        author_publications_dict[each] = 1
    return author_publications_dict

def reverse_dict(dict_in):
    dict_out = {}
    for key in dict_in.keys():
        value = dict_in[key]
        if not isinstance(value, list):
            if value in dict_out:
                dict_out[value].append(key)
            else:
                dict_out[value] = [key]
        else:
            for each in value:
                dict_out[each] = key
    return dict_out

def get_top_values(dict_in, rank):
    top_values_list = []
    top_rank = sorted(list(dict_in.keys()), reverse = True)[:rank]
    for each in top_rank:
       top_values_list.extend(dict_in[each])
    return top_values_list

def del_invalid_keys(dict_in, invalid_keys_list):
    for each in invalid_keys_list:
        del dict_in[each]
    return dict_in

def venue_publications(path):
    venue_publications_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#c"):
                line = line.lstrip("#c").rstrip("\n").strip()
                if line in venue_publications_dict:
                    venue_publications_dict[line] += 1
                else:
                    venue_publications_dict[line] = 1
    return venue_publications_dict

def references_citations(path):
    references_dict, citations_dict = {}, {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#*"):
                line = line.lstrip("#*").rstrip("\n")
                publication = line.strip()
                references_dict[publication] = 0
            if line.startswith("#%"):
                references_dict[publication] += 1
                line = line.lstrip("#%").rstrip("\n").strip()
                if line in citations_dict:
                    citations_dict[line] += 1
                else:
                    citations_dict[line] = 1
    return references_dict, citations_dict

def citation_to_publication(path):
    cite_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#*"):
                line = line.lstrip("#*").rstrip("\n")
                publication = line.strip()
            if line.startswith("#index"):
                line = line.lstrip("#index").rstrip("\n").strip()
                cite_dict[line] = publication
    return cite_dict

def publication_to_citation(path):
    title_to_cite_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#*"):
                line = line.lstrip("#*").rstrip("\n")
                publication = line.strip()
            if line.startswith("#index"):
                line = line.lstrip("#index").rstrip("\n").strip()
                title_to_cite_dict[publication] = line
    return title_to_cite_dict

def plot_references_citations():
    ref_dict, cite_dict = references_citations("acm.txt")
    plt.style.use('_mpl-gallery')
    # make data
    x1, x2 = np.array(list(ref_dict.values())), np.array(list(cite_dict.values()))
    # plot:
    binwidth = 10
    plt.subplot(2, 1, 1)
    plt.hist(x1, bins = range(min(x1), max(x1) + binwidth, binwidth),
             label = "references", log = True)
    plt.xlabel('number of references')
    plt.ylabel('number of publications')
    plt.title(' number of references per publication')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    binwidth = 100
    plt.hist(x2, bins = range(min(x2), 10**4 + binwidth, binwidth),
             label = "citations", log = True)
    plt.xlabel('number of citations')
    plt.ylabel('nnumber of publications')
    plt.title(' number of citations per publication')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    # Print out top reference publication and top citation publication
    rev_r_dict, rev_c_dict = reverse_dict(ref_dict), reverse_dict(cite_dict)
    top_ref, top_cite = (get_top_values(rev_r_dict, 1)[0],
                         get_top_values(rev_c_dict, 1)[0])
    top_cite = citation_to_publication("acm.txt")[top_cite]
    print(f"{top_ref} has the largest number of references, "\
          f"{top_cite} has the largest number of citations.")

def index_venue(path):
    index_venue_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#c"):
                venue = line.lstrip("#c").rstrip("\n").strip()
            if line.startswith("#index"):
                index = line.lstrip("#index").rstrip("\n").strip()
                index_venue_dict[index] = venue    
    return index_venue_dict

def venue_citations(path):
    venue_citations_dict, index_venue_dict = {}, index_venue(path)
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#%"):
                citation = line.lstrip("#%").rstrip("\n").strip()
                venue = index_venue_dict[citation]
                if venue in venue_citations_dict:
                    venue_citations_dict[venue] += 1
                else:
                    venue_citations_dict[venue] = 1  
    return venue_citations_dict

def venue_impact():
    venue_publications_dict = venue_publications("acm.txt")
    venue_citations_dict = venue_citations("acm.txt")
    venue_impact_dict = {}
    for venue in venue_publications_dict:
        if venue in venue_citations_dict:
            impact_factor = venue_citations_dict[venue] / venue_publications_dict[venue]
            venue_impact_dict[venue] = impact_factor
        else:
            venue_impact_dict[venue] = 0
    return venue_impact_dict

def plot_venue_impact():
    venue_impact_dict = venue_impact()
    plt.style.use('fivethirtyeight')
    # make data
    x = np.array(list(venue_impact_dict.values()))
    # plot:
    plt.hist(x, bins = 50, log = True)
    plt.xlabel('impact factor')
    plt.ylabel('number of venues')
    plt.title('impact factor per venue')
    plt.tight_layout()
    plt.show()
    # Print out top reference publication and top citation publication
    rev_dict = reverse_dict(venue_impact_dict)
    top_impact = get_top_values(rev_dict, 1)[0]
    top_value = venue_impact_dict[top_impact]
    print(f"{top_impact} has the highest apparent impact factor of {top_value}.")

def adjust_venues(path):
    venue_publications_dict = venue_publications(path)
    venue_citations_dict = venue_citations(path)
    rev_dict = reverse_dict(venue_publications_dict)
    invalid_list = [i for i in range(10) if i in rev_dict]
    rev_dict = del_invalid_keys(rev_dict, invalid_list)
    adjusted_publications = reverse_dict(rev_dict)
    invalid_keys = [key for key in venue_citations_dict if key not in adjusted_publications]
    adjusted_cites = del_invalid_keys(venue_citations_dict, invalid_keys)
    return adjusted_publications, adjusted_cites

def adjust_venue_impact():
    venue_publications_dict, venue_citations_dict = adjust_venues("acm.txt")
    venue_impact_dict = {}
    for venue in venue_publications_dict:
        if venue in venue_citations_dict:
            impact_factor = venue_citations_dict[venue] / venue_publications_dict[venue]
            venue_impact_dict[venue] = impact_factor
        else:
            venue_impact_dict[venue] = 0
    return venue_impact_dict

def plot_adjust_venue_impact():
    venue_impact_dict = adjust_venue_impact()
    plt.style.use('_mpl-gallery')
    # make data
    x = np.array(list(venue_impact_dict.values()))
    # plot:
    #binwidth = 0.02
    plt.hist(x, bins = 50, log = True)
    plt.xlabel('impact factor')
    plt.ylabel('number of venues')
    plt.title('adjusted impact factor per venue')
    plt.tight_layout()
    plt.show()
    # Print out top reference publication and top citation publication
    rev_dict = reverse_dict(venue_impact_dict)
    top_impact = get_top_values(rev_dict, 1)[0]
    top_value = np.round(venue_impact_dict[top_impact], 2)
    print(f"{top_impact} has the highest apparent impact factor of {top_value}.")

def venue_publication_indices(path):
    venue_indices_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#c"):
                venue = line.lstrip("#c").rstrip("\n").strip()
            if line.startswith("#index"):
                index = line.lstrip("#index").rstrip("\n").strip()
                if venue in venue_indices_dict:
                    venue_indices_dict[venue].append(index)
                else:
                    venue_indices_dict[venue] = [index]
    return venue_indices_dict

def publication_citations_count(venue):
    title_cites_count_dict = {}
    venue_indices_dict = venue_publication_indices("acm.txt")
    #title_to_cite_dict = publication_to_citation("acm.txt")
    citations_dict = references_citations("acm.txt")[1]
    publication_indices = venue_indices_dict[venue]
    for each in publication_indices:
        try:
            cites_count = citations_dict[each]
            title_cites_count_dict[each] = cites_count
        except:
            title_cites_count_dict[each] = 0
    return title_cites_count_dict

def impact_factor_to_median():
    title_cites_count_dict = publication_citations_count('INFORMS Journal on Computing')
    data = np.array(list(title_cites_count_dict.values()))
    impact_factor = np.mean(data)
    cites_median = np.percentile(data, 50)
    print("The impact factor is %.2f, the median number of %.2f."
          % (impact_factor, cites_median))

def year_publications(path):
    year_publications_dict = {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#t"):
                year = line.lstrip("#t").rstrip("\n").strip()
            if line.startswith("#index"):
                publication = line.lstrip("#index").rstrip("\n").strip()
                if year in year_publications_dict:
                    year_publications_dict[year].append(publication)
                else:
                    year_publications_dict[year] = [publication]
    return year_publications_dict

def indices_citations(path):
    references_dict, citations_dict = {}, {}
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if line.startswith("#index"):
                line = line.lstrip("#index").rstrip("\n")
                publication = line.strip()
                references_dict[publication] = 0
            if line.startswith("#%"):
                references_dict[publication] += 1
                line = line.lstrip("#%").rstrip("\n").strip()
                if line in citations_dict:
                    citations_dict[line] += 1
                else:
                    citations_dict[line] = 1
    return references_dict, citations_dict

def plot_refers_cites_year():
    year_refers_dict, year_cites_dict = {}, {}
    year_publications_dict = year_publications("acm.txt")
    ref_dict, cite_dict = indices_citations("acm.txt")
    for each in year_publications_dict:
        total_refs = total_cites = 0 
        for title in year_publications_dict[each]:
            total_refs += ref_dict[title]
            try:
                cites = cite_dict[title]
                total_cites += cites
            except:
                total_cites += 0
        avg_refs = total_refs / len(year_publications_dict[each])
        avg_cites = total_cites / len(year_publications_dict[each])
        year_refers_dict[each] = avg_refs
        year_cites_dict[each] = avg_cites
    # Plot
    plt.style.use('_mpl-gallery')
    x1 = np.array(list(year_refers_dict.keys()))
    y1 = np.array(list(year_refers_dict.values()))
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, label = "references")
    plt.xticks([])
    #plt.yscale('log')
    plt.legend(loc = 'upper right')
    plt.xlabel('publication year')
    plt.ylabel('avg references per publication')
    plt.title('yearly avg references per publication change')
    plt.tight_layout()
    x2 = np.array(list(year_cites_dict.keys()))
    y2 = np.array(list(year_cites_dict.values()))
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, label = "citations")
    plt.xticks([])
    #plt.yscale('log')
    plt.legend(loc = 'upper right')
    plt.xlabel('publication year')
    plt.ylabel('avg citations per publication')
    plt.title('yearly avg citations per publication change')
    plt.tight_layout()
    plt.show()
