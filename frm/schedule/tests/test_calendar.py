# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())


from frm.schedule.calendar import get_calendar


holiday_calendar = get_calendar(ccys=['usd','aud'])

    