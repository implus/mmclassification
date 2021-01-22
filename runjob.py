import os, sys, random, time

if len(sys.argv) < 2:
    print('python xxx.py work_dirs/slurmjobs/xxxx')
    exit()

slurmjob = sys.argv[1]
prefix = slurmjob.split('/')[-1]
exit_flag = False

while True:
    jobname = prefix + '-%03d' % random.randint(0, 900)
    cmd = f'slurm batch {slurmjob} --job-name={jobname}'
    print(f'run cmd: {cmd}')
    os.system(cmd)
    time.sleep(10)

    while True:
        os.system('rm -rf tmp.txt')
        ckcmd = f'slurm queue | grep {jobname} > tmp.txt'
        print(f'run ckcmd: {ckcmd}')
        os.system(ckcmd)

        time.sleep(30)

        fi = open('tmp.txt').readlines()
        if len(fi) == 0:
            break
        else:
            v = fi[0].split(' ')
            nv = []
            for x in v:
                if len(x) > 0:
                    nv.append(x)
            print(nv)
            print(nv[5])
            jobid = nv[0].split('(')[0]
            print(f'jobid = {jobid}')
            if '-5m' in nv[5]:
                cmd = f'slurm cancel {jobid}'
                print(f'run cancel cmd: {cmd}')
                os.system(cmd)
                time.sleep(10)
                break
            elif '5m' in nv[5]:
                exit()
            else:
                continue


