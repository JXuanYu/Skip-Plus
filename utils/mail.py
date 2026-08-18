import yagmail


class MailClient(object):
    def __init__(self, cfg):
        self.enabled = cfg['username'] != 'disabled' and cfg['to'] != 'disabled'
        self.to = cfg['to']

        if self.enabled:
            self.client = yagmail.SMTP(cfg['username'], cfg['password'], cfg['host'])
        else:
            self.client = None

    def send(self, subject, contents):
        if self.enabled and self.client is not None:
            self.client.send(self.to, subject, contents, prettify_html=False)
