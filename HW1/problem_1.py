'''
    DS 5230
    Summer 2022
    HW1_Problem_1_Aminer_basic_dataset_analysis

    Conduct basic dataset analysis to the Aminer dataset

    Hongyan Yang
'''


from problem_1_drivers import *

def main(question):
    if question == "A":
        distinct_items = extract_attributes("acm.txt")

    elif question == "B":
        related_venues = look_up_venue("acm.txt")
        print(related_venues[1])

    elif question == "C":
        # check data
        author_publications_dict = plot_author_publications("acm.txt")
        publications_author_dict = reverse_dict(author_publications_dict)
        authors_list = get_top_values(publications_author_dict, 20)
        print(authors_list)
        author_publications_dict = author_publications("acm.txt")
        print(author_publications_dict["Virgil D. Gligor"])
        plt.style.use('_mpl-gallery')
        # clean data
        invalid_keys_list = ['Jr.', '-', 'III', 'II']
        adjusted_dict = del_invalid_keys(plot_author_publications("acm.txt"),
                                         invalid_keys_list)
        x = np.array(list(adjusted_dict.values()))
        # plot:
        binwidth = 100
        plt.hist(x, bins = range(min(x), max(x) + binwidth, binwidth), log = True)
        plt.xlabel('number of publications')
        plt.ylabel('number of authors')
        plt.title(' number of publications per author')
        plt.tight_layout()
        plt.show()

    elif question == "D":
        invalid_keys_list = ['Jr.', '-', 'III', 'II']
        adjusted_dict = del_invalid_keys(plot_author_publications("acm.txt"),
                                         invalid_keys_list)
        x = np.array(list(adjusted_dict.values()))
        print("mean of publications per author: ", np.round(np.mean(x), 2))
        print("std of publications per author: ", np.round(np.std(x), 2))
        print("Q1 of publications per author: ", np.percentile(x, 25))
        print("median of publications per author: ", np.percentile(x, 50))
        print("Q3 of publications per author: ", np.percentile(x, 75))

    elif question == "E":
        venue_publications_dict = venue_publications("acm.txt")
        plt.style.use('_mpl-gallery')
        # make data
        x = np.array(list(venue_publications_dict.values()))
        # plot:
        binwidth = 100
        plt.hist(x, bins = range(min(x), max(x) + binwidth, binwidth), log = True)
        plt.xlabel('number of publications')
        plt.ylabel('number of venues')
        plt.title('number of publications per venue')
        plt.tight_layout()
        plt.show()
        print("mean of publications per venue: ", np.round(np.mean(x)))
        print("std of publications per venue: ", np.round(np.std(x)))
        print("Q1 of publications per venue: ", np.percentile(x, 25))
        print("median of publications per venue: ", np.percentile(x, 50))
        print("Q3 of publications per venue: ", np.percentile(x, 75))
        dict_out = reverse_dict(venue_publications_dict)
        top_values_list = get_top_values(dict_out, 1)
        print(f"Venue {top_values_list[0]} has the largest number of publications.")

    elif question == "F":
        plot_references_citations()

    elif question == "G":
        plot_venue_impact()

    elif question == "I":
        plot_adjust_venue_impact()
        venue_indices_dict = venue_publication_indices("acm.txt")
        print(len(venue_indices_dict["INFORMS Journal on Computing"]))
        impact_factor_to_median()

    elif question == "J":
        plot_refers_cites_year()

       
if __name__ == "__main__":
    #pass
    main("J")
