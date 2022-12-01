# Tejas Sharma
# Note: please install the requests library before running the code
# pip install requests
import requests
from collections import defaultdict
import datetime

def get_partners():
    """
    :return: An array of all the partners
    """
    # Query the API
    url = 'https://candidate.hubteam.com/candidateTest/v3/problem/dataset?userKey=64daed58fe7c01f152cd78bbd30d'
    res = requests.get(url)

    # Ensure the request was successful
    if res.status_code == 200:
        return res.json()['partners']
    else:
        return []

def group_partners_by_country(partners):
    """
    Given a list of partners, group them by country using a hashmap
    :return: A hashmap of country -> partners in that country
    """

    # Hashmap that defaults to an empty list if the key isn't present
    partners_per_country = defaultdict(list)

    for partner in partners:
        # Add this partner to the country they are located in
        partner_country = partner["country"]
        partners_per_country[partner_country].append(partner)

    return partners_per_country

def find_best_available_day_for_partners(partners):
    """
    Given partners, find the best available day for a 2-day period
    :param partners: list of partners
    :return: the date that accommodates for the largest number of partners
    """

    # A dict of day -> # of partners available that day
    partners_per_day = defaultdict(int)
    # A dict of day -> the email address of the partners available that day
    emails_of_partners_per_day = defaultdict(list)

    # For each partner
    for partner in partners:
        # For each available start date
        for date in partner["availableDates"]:
            # Parse the date into a python date
            parsed_date = datetime.datetime.fromisoformat(date)
            # Get the next day as a python date
            day_after_parsed_date = parsed_date + datetime.timedelta(days = 1)
            # Convert it to a string
            day_after_parsed_date_str = str(day_after_parsed_date).split(" ")[0]
            # Check if the partner is available the next day
            if day_after_parsed_date_str in partner["availableDates"]:
                # This day works for the partner, increment the # of partners available this day
                partners_per_day[date] += 1
                emails_of_partners_per_day[date].append(partner["email"])

    # If no best day was found, partners_per_day will be empty
    if not partners_per_day:
        return None, 0, []

    # Otherwise, get the day that has the most number of partners available
    best_day = max(partners_per_day, key=partners_per_day.get)

    # Return the best day, the # of partners available that day, and their contacts
    return best_day, partners_per_day[best_day], emails_of_partners_per_day[best_day]

def create_post_request_body(partners_per_country):
    """
    Generate the body for the POST request
    :param partners_per_country: the partners grouped by country
    :return: the body as a python dict object
    """
    body = {
        "countries": []
    }
    for country in partners_per_country:
        best_day, attendeeCount, attendees = find_best_available_day_for_partners(partners_per_country[country])
        body["countries"].append({
            "attendeeCount": attendeeCount,
            "attendees": attendees,
            "name": country,
            "startDate": best_day
        })
    return body

def send_post_request(post_request_body):
    """
    Given the request body, send a post request
    :param post_request_body: The POST request body
    :return: None
    """
    url = 'https://candidate.hubteam.com/candidateTest/v3/problem/result?userKey=64daed58fe7c01f152cd78bbd30d'
    res = requests.post(url, json=post_request_body)
    if res.status_code == 200:
        print('Response successfully sent and received status code 200')
    else:
        print('Response sent but received status code ', res.status_code)

# The main function that will be run
def main():
    partners = get_partners()
    partners_per_country = group_partners_by_country(partners)
    post_request_body = create_post_request_body(partners_per_country)
    send_post_request(post_request_body)

if __name__ == "__main__":
    main()