import wikipedia
import sys
import codecs

class Wikipedia:

    def __init__(self):
        pass

    def get_wikipedia_page(self,species_name):
        sharks=wikipedia.page(species_name)
        with open("cheetahs.txt",'w') as f:
            c=sharks.content.encode('utf-8')
            f.write(str(c))


w=Wikipedia()
w.get_wikipedia_page("cheetah")