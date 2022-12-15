import gspread

class Spreadsheet:
    def __init__(self, spreadsheet_name=None, worksheet=None):
        self.gc = gspread.oauth()
        if spreadsheet_name is not None:
            self.sh = self.gc.open(spreadsheet_name)
        else:
            self.sh = None

        if worksheet is not None:
            self.ws = self.sh.worksheet(worksheet)
        else:
            self.ws = None

    def open_spreadsheet(self, spreadsheet_name):
        self.sh = self.gc.open(spreadsheet_name)
    
    def open_worksheet(self, worksheet):
        self.ws = self.gc.worksheet(worksheet)
    
    def get_data(self, range_or_val, worksheet=None):
        if worksheet is not None:
            data = self.sh.worksheet(worksheet).get(range_or_val)
        else:
            data = self.ws.get(range_or_val)
        return data
    
    def write_to_worksheet(self, range, values, worksheet=None, value_input_option='USER_ENTERED'):
        if worksheet is not None:
            self.ws.update(
                range, 
                values, 
                value_input_option=value_input_option
            )
        else:
            self.sh.worksheet(
                worksheet
            ).update(
                range, 
                values, 
                value_input_option=value_input_option
            )
