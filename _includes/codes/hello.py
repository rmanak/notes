###### Comment #######
def clean_text(text,replace_dict=None,words_replacer=None):

    text = text.lower()

    if replace_dict is not None:
        for k, v in replace_dict.items():
            text = re.sub(k,v,text)

    if words_replacer is not None:
        text = text.split(' ')
        text = words_replacer.replace(text)
        text = ' '.join(text)

    return text
