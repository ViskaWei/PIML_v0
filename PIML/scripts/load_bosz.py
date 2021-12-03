from PIML.util.basebox import BaseBox

def main():
    b = BaseBox()
    Res = 5000
    _ = b.IO.load_laszlo_bosz(Res, getPara=True, save=1, overwrite=True)



if __name__ == "__main__":
    main()
