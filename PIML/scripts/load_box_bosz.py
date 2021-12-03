from PIML.util.basebox import BaseBox

def main():
    b = BaseBox()
    Res = 5000
    wave, flux, pdx, para = b.IO.load_bosz(Res)
    b.box_data(wave, flux, pdx, para, DBdx=None, Res=Res, Rs=None, out=0, save=True)



if __name__ == "__main__":
    main()

