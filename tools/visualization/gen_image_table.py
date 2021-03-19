import argparse
import math
import os
import os.path as osp

from PIL import Image


class Element:
    """A data element of a row in a table."""

    def __init__(self, htmlCode=''):
        self.htmlCode = htmlCode
        self.isHeader = False
        self.drawBorderColor = ''

    def imgToHTML(self, img_path, width=200, overlay_path=None):
        res = '<img src="' + img_path.strip().lstrip() + '" width="' + str(
            width) + 'px" '
        if self.drawBorderColor:
            res += 'style="border: 10px solid ' + self.drawBorderColor + '" '
        if overlay_path:
            res += 'onmouseover="this.src=\'' + overlay_path.strip().lstrip(
            ) + '\';"'
            res += 'onmouseout="this.src=\'' + img_path.strip().lstrip(
            ) + '\';"'
        res += '/>'
        return res

    def addImg(self,
               img_path,
               width=400,
               imsize=None,
               overlay_path=None,
               scale=None,
               out=None):
        # bboxes must be a list of [x,y,w,h] (i.e. a list of lists)
        # imsize is the natural size of image at img_path.. used for putting bboxes, not required otherwise
        # even if it's not provided, I'll try to figure it out -- using the typical use cases of this software
        # overlay_path is image I want to show on mouseover
        assert osp.exists(img_path), img_path
        self.htmlCode += self.imgToHTML(
            osp.relpath(img_path, out), width, overlay_path)

    def addTxt(self, txt):
        if self.htmlCode:  # not empty
            self.htmlCode += '<br />'
        self.htmlCode += str(txt)

    def getHTML(self):
        return self.htmlCode

    def setIsHeader(self):
        self.isHeader = True

    def setDrawCheck(self):
        self.drawBorderColor = 'green'

    def setDrawUnCheck(self):
        self.drawBorderColor = 'red'

    def setDrawBorderColor(self, color):
        self.drawBorderColor = color

    @staticmethod
    def getImSize(impath):
        im = Image.open(impath)
        return im.size


class TableRow:

    def __init__(self, isHeader=False, rno=-1):
        self.isHeader = isHeader
        self.elements = []
        self.rno = rno

    def addElement(self, element):
        self.elements.append(element)

    def getHTML(self):
        html = '<tr>'
        if self.rno >= 0:
            html += '<td><a href="#' + str(self.rno) + '">' + str(
                self.rno) + '</a>'
            html += '<a name=' + str(self.rno) + '></a></td>'
        for e in self.elements:
            if self.isHeader or e.isHeader:
                elTag = 'th'
            else:
                elTag = 'td'
            html += '<%s>' % elTag + e.getHTML() + '</%s>' % elTag
        html += '</tr>'
        return html


class Table:

    def __init__(self, rows=[]):
        self.rows = [row for row in rows if not row.isHeader]
        self.headerRows = [row for row in rows if row.isHeader]

    def addRow(self, row):
        if not row.isHeader:
            self.rows.append(row)
        else:
            self.headerRows.append(row)

    def getHTML(self, ):
        html = '<table border=1 id="data">'
        for r in self.headerRows + self.rows:
            html += r.getHTML()
        html += '</table>'
        return html

    def countRows(self):
        return len(self.rows)


class TableWriter:

    def __init__(self,
                 table,
                 outputdir,
                 rowsPerPage=20,
                 pgListBreak=20,
                 desc=''):
        self.outputdir = outputdir
        self.rowsPerPage = rowsPerPage
        self.table = table
        self.pgListBreak = pgListBreak
        self.desc = desc

    def write(self):
        os.makedirs(self.outputdir, exist_ok=True)
        nRows = self.table.countRows()
        pgCounter = 1
        for i in range(0, nRows, self.rowsPerPage):
            rowsSubset = self.table.rows[i:i + self.rowsPerPage]
            t = Table(self.table.headerRows + rowsSubset)
            f = open(
                os.path.join(self.outputdir,
                             str(pgCounter) + '.html'), 'w')
            pgLinks = self.getPageLinks(
                int(math.ceil(nRows * 1.0 / self.rowsPerPage)), pgCounter,
                self.pgListBreak)

            f.write(pgLinks)
            f.write('<p>' + self.desc + '</p>')
            f.write(t.getHTML())
            f.write(pgLinks)
            f.close()
            pgCounter += 1

    @staticmethod
    def getPageLinks(nPages, curPage, pgListBreak):
        if nPages < 2:
            return ''
        links = ''
        for i in range(1, nPages + 1):
            if not i == curPage:
                links += '<a href="' + str(i) + '.html">' + str(
                    i) + '</a>&nbsp'
            else:
                links += str(i) + '&nbsp'
            if (i % pgListBreak == 0):
                links += '<br />'
        return '\n' + links + '\n'


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('out', help='out put directory')
    parser.add_argument('input', help='input_directory, should under out')
    args = parser.parse_args()
    return args


def main():
    table = Table()
    args = parse_args()
    dirs = []
    files = []
    for d in sorted(os.listdir(args.input)):
        cur_files = []
        if osp.isdir(osp.join(args.input, d)):
            dirs.append(d)
            for f in sorted(os.listdir(osp.join(args.input, d))):
                cur_files.append(osp.join(args.input, d, f))
        files.append(cur_files)
    num_cols = len(files)

    # Header
    header = TableRow(isHeader=True)
    for i in range(num_cols + 1):
        e = Element()
        if i == 0:
            e.addTxt('index')
        else:
            e.addTxt(osp.basename(dirs[i - 1]))
        header.addElement(e)
    table.addRow(header)

    num_rows = min(len(fs) for fs in files)
    for i in range(num_rows):
        if i % 5 == 0:
            info_r = TableRow(rno=i)
            for ii in range(num_cols):
                e = Element()
                e.addTxt(osp.basename(dirs[ii]))
                info_r.addElement(e)
            table.addRow(info_r)

        r = TableRow(rno=i)
        for j in range(num_cols):
            e = Element()
            e.addImg(files[j][i], out=args.out)
            r.addElement(e)
        table.addRow(r)

    writer = TableWriter(table, args.out)
    writer.write()


if __name__ == '__main__':
    main()
