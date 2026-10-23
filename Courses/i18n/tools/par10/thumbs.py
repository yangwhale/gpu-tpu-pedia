import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
d, w = sys.argv[1], sys.argv[2]; sc = float(sys.argv[3]) if len(sys.argv)>3 else 0.25
Z = Image.open(f'{d}/full-zh-{w}.png'); E = Image.open(f'{d}/full-en-{w}.png')
z = Z.resize((int(Z.width*sc), int(Z.height*sc))); e = E.resize((int(E.width*sc), int(E.height*sc)))
H = max(z.height, e.height); seg = 2400; n = 0
for y in range(0, H, seg):
    im = Image.new('RGB', (z.width+e.width+20, seg), 'white')
    im.paste(z.crop((0, y, z.width, min(y+seg, z.height))) if y < z.height else Image.new('RGB',(1,1),'white'), (0,0))
    im.paste(e.crop((0, y, e.width, min(y+seg, e.height))) if y < e.height else Image.new('RGB',(1,1),'white'), (z.width+20,0))
    im.save(f'{d}/thumb-{w}-{n:02d}.png'); n += 1
print(n, 'thumbs', z.size, e.size)
