import re

class Tokeniser:
    def __init__(self,s):
        self.st=s
    def replace_url(self,placeholder="<URL>"):
        url_pattern=re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
        result_text = re.sub(url_pattern, placeholder, self.st)
        return result_text
    def replace_mailid(self,placeholder="<MAILID>"):
        mailid_pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        result_text=re.sub(mailid_pattern,placeholder,self.st)
        return result_text
    def replace_hashtags(self,placeholder="<HASHTAG>"):
        # hash_pattern=re.compile(r'\s#([a-zA-Z0-9_]+)')
        hash_pattern=re.compile(r'(#+[a-zA-Z0-9(_)]{1,})')
        # assuming # can be followed by a-z (cap/small), digits 0-9 and underscore(_) as well
        result_text = re.sub(hash_pattern, placeholder, self.st)
        return result_text
    def replace_mentions(self,placeholder="<MENTION>"):
        mention_pattern=re.compile(r'@([a-zA-Z0-9_]+)')
        # assuming @ can be followed by a-z (cap/small), digits 0-9 and underscore(_) as well
        result_text = re.sub(mention_pattern, placeholder, self.st)
        return result_text
    def replace_num(self,placeholder="<NUM>"):
        num_pattern=re.compile(r'\d+')
        result_text=re.sub(num_pattern,placeholder,self.st)
        return result_text
    def convert(self):
        self.st=self.replace_url()
        self.st=self.replace_mailid()
        self.st=self.replace_hashtags()
        self.st=self.replace_mentions()
        self.st=self.replace_num()
        pattern = re.compile(r'([A-Z])\.')
        self.st=re.sub(pattern,lambda match: match.group(1),self.st)
        pattern = re.compile(r'(Mr)\.')
        self.st=re.sub(pattern,'Mr',self.st)
        pattern = re.compile(r'(Ms)\.')
        self.st=re.sub(pattern,'Ms',self.st)
        pattern = re.compile(r'(Mrs)\.')
        self.st=re.sub(pattern,'Mrs',self.st)
        pattern = re.compile(r'(Dr)\.')
        self.st=re.sub(pattern,'Dr',self.st)
        sentences=re.split(r'[.|!|?]+|\n\n',self.st) # splitting into sentences when 2 consecutive new line characters, full stop, exclamation or question mark
        ret=[]
        for a in sentences:
            z=re.findall(r'<\w+>|\w+|[,;\'\"/^&%$-:+=*\(\)\[\]\{\}]',a)
            if z:
                ret.append(z)
        return ret

# s="The Project Gutenberg eBo&ok, #567,Pride and (Prejudice), @Aks_123/by Jane $Austen, http://go.com, 56 times! Edited by R. W. (Robert William) Chapman This eBook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever.  You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this eBook or online at www.gutenberg.org Title: Pride and Prejudice"
# with open('Pride and Prejudice.txt', "r") as file:
#     s = file.read()
if __name__=="__main__":
    s=input("your text : ")
    tok=Tokeniser(s)
    ans=tok.convert()
    print(f'tokenized text: {ans}')