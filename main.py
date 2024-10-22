import torch.optim
from torch.nn import BCELoss
import time
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from utils import *
from model import *


def main():
    args = parse_arguments()
    ngpu = torch.cuda.device_count()
    if ngpu > 0:
        print("Running on gpu..\n")
        device = "cuda"
    else:
        print("Running on cpu..\n")
        device = "cpu"

    print(f"Loading data {args.data}..")
    X, Y = load_data(args.data)
    print(f"The shape of data is {X.shape}.")
    print(f"The value of data is between {X.min()} and {X.max()}.\n")

    torch.cuda.empty_cache()

    latent_size = X.shape[1]

    weight_decay_E = 1e-3
    weight_decay_G = 1e-3

    netE = Estimator(ngpu, latent_size).to(device)
    print("Confidence Estimators:")
    print(netE)

    criterionE = lossE
    optimizerE = optim.Adam(netE.parameters(), lr=args.lr_E, weight_decay=weight_decay_E)

    history = {
        'errE': [],
        'auc': []
    }

    k = args.k

    if args.model == 'ACE':
        history['rec_err'] = []
        history['con_err'] = []
    if args.model == 'GCN-UOD':
        netE.load_state_dict(torch.load(f'weights/ACE_{args.data}.pt'))
        netG = []
        optimizerG = []
        for i in range(k):
            netG.append(Generator(ngpu, latent_size).to(device))
            optimizerG.append(torch.optim.Adam(netG[i].parameters(), lr=args.lr_G, weight_decay=weight_decay_G))
            print(f"Data Generator {i}:\n{netG[i]}")

        guidance_criterion = BCELoss()
        criterionG = BCELoss()

        history['errG'] = []

    if args.model == 'ACE':
        batch_size = X.shape[0] // 12
        if args.data == 'Shuttle':
            batch_size = X.shape[0] // 10
    else:
        batch_size = X.shape[0]

    torch_data = TensorDataset(X)
    dataloader = DataLoader(torch_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    start_time = time.time()

    for epoch in range(args.epochs):
        sum_errE = 0.
        sum_rec_err = 0.
        sum_con_err = 0.
        sum_errG = 0.
        batch_num = len(dataloader)

        netE.train()

        for index, batch_x in enumerate(dataloader, 0):

            if args.model == 'ACE':
                netE.zero_grad()
                real_data = batch_x[0].to(device)
                rec_data, conf = netE(add_noise(real_data))
                errE, rec_err, con_err = criterionE(rec_data, conf, real_data, args.alpha, device)

                optimizerE.zero_grad()
                errE.backward()
                optimizerE.step()

                sum_errE += errE.item()
                sum_rec_err += rec_err.item()
                sum_con_err += con_err.item()

            elif args.model == 'GCN-UOD':
                errG = []

                _, real_conf = netE(X.to(device).detach())
                real_conf = pd.DataFrame(real_conf.cpu().detach().numpy())
                mean_conf = np.mean(real_conf).item()

                for i in range(k):
                    netG[i].zero_grad()
                    noise = torch.randn(size=(batch_size, latent_size), device=device)
                    generated_data = netG[i](noise)
                    target = real_conf.quantile(i / k).item()
                    label = torch.full((batch_size,), target, dtype=torch.float, device=device)
                    _, conf = netE(generated_data)
                    conf = conf.view(-1)
                    errG.append(criterionG(conf, label))

                    optimizerG[i].zero_grad()
                    errG[i].backward(retain_graph=True)
                    optimizerG[i].step()

                errG = torch.as_tensor(errG)
                errG = torch.mean(errG)
                sum_errG = sum_errG + errG.item()

                netE.zero_grad()
                generated_data, label = [], []
                noise_size = batch_size
                noise = torch.randn(size=(noise_size, latent_size), device=device)
                block = ((1 + k) * k) // 2

                for i in range(k):
                    noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                    noise_end = int((((k + (k - i)) * (i + 1)) / 2) * (noise_size // block)) if i != (k - 1) else batch_size
                    noise_i = torch.as_tensor(noise[noise_start: noise_end].type(torch.float32))

                    target = real_conf.quantile(i / k).item()
                    if target < mean_conf:
                        target = 0.5 * target
                    else:
                        target = 2 * target

                    if args.drop_one:
                        if target < 1.:
                            generated_data.append(netG[i](noise_i))
                            label.append(torch.full((noise_i.size(0),), target, dtype=torch.float, device=device))
                    else:
                        if target > 1.:
                            target = 1.

                        generated_data.append(netG[i](noise_i))
                        label.append(torch.full((noise_i.size(0),), target, dtype=torch.float, device=device))

                generated_data = torch.cat(generated_data)
                label = torch.cat(label)

                _, gen_conf = netE(add_noise(generated_data))
                gen_conf = gen_conf.view(-1)
                errE = guidance_criterion(gen_conf, label)

                optimizerE.zero_grad()
                errE.backward()
                optimizerE.step()

                sum_errE += errE.item()

            if args.model == 'ACE':
                print(
                    f"Training: [{epoch + 1}/{args.epochs}][{index + 1}/{batch_num}]  E Loss: {errE.item():.4f}, "
                    f"Rec. Loss: {rec_err.item():.4f}, Con. Loss: {con_err.item():.4f}"
                )
            elif args.model == 'GCN-UOD':
                print(
                    f"Training: [{epoch + 1}/{args.epochs}][{index + 1}/{batch_num}] G Loss: {errG.item():.4f}, "
                    f"E Loss: {errE.item():.4f}"
                )

        mean_errE = sum_errE / batch_num
        history['errE'].append(mean_errE)

        if args.model == 'ACE':
            mean_rec_err = sum_rec_err / batch_num
            mean_con_err = sum_con_err / batch_num
            history['rec_err'].append(mean_rec_err)
            history['con_err'].append(mean_con_err)
        elif args.model == 'GCN-UOD':
            mean_errG = sum_errG / batch_num
            history['errG'].append(mean_errG)

        netE.eval()

        with torch.no_grad():
            _, conf = netE(X.to(device))

            outlier_score = conf.view(-1).cpu().detach().numpy()

            auc_score = roc_auc_score(Y.cpu().detach().numpy(), outlier_score)
            history['auc'].append(auc_score)

        if args.model == 'ACE':
            print(
                f"\nTesting: [{epoch + 1}/{args.epochs}]  E Loss: {mean_errE:.4f}, Rec. Loss: {mean_rec_err:.4f}, "
                f"Con. Loss: {mean_con_err:.4f}, AUC: {auc_score:.4f}\n"
            )
        elif args.model == 'GCN-UOD':
            print(
                f"\nTesting: [{epoch + 1}/{args.epochs}] G Loss: {mean_errG:.4f}, E Loss: {mean_errE:.4f}, "
                f"AUC: {auc_score:.4f}\n"
            )

    end_time = time.time()
    running_time = end_time - start_time
    print(f"Model training and testing have been completed. The process took {running_time:.4f} s.")

    model_info = {
        'model': netE,
        'model_name': args.model,
        'data_name': args.data,
        'time': running_time
    }
    save_results(history, outlier_score, Y.cpu().detach().numpy(), model_info)


if __name__ == "__main__":
    main()
