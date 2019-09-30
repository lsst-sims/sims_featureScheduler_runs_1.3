

# write the script to run maf on everything
# find . -name "*.db" -print > filelist

script = open('maf_script', 'w')
fl = open('filelist', 'r')
for line in fl:
    script.write('python ../glance_dir.py --db .%s' % line)
    script.write('python ../scimaf_dir.py --db .%s' % line)

fl.close()
script.close()
