


class ArchaicEarlyStopper():
    def __init__(self, patience_eval=10, patience_dL=50, initLoss = 1e6):
        """

        Args:
            patience_eval (int, optional): [description]. Defaults to 10.
            patience_dL (int, optional): [description]. Defaults to 50.
            initLoss ([type], optional): [description]. Defaults to 1e6.
        """
        self.patience_eval = patience_eval
        self.patience_dL = patience_dL
        self.latent_eval_loss = initLoss
        self.latent_train_loss = initLoss
        self.latent_delta_loss = initLoss
        self.eval_sucsincr = 0
        self.train_sucsincr = 0
        self.delta_sucsincr = 0
        self.deltaList = []
        self.evalList = []
        self.doStop = False
        self.evalStop = False
        self.deltaStop = False
        
    def step(self, current_eval_loss, current_train_loss):
        # eval loss
        if current_eval_loss < self.latent_eval_loss:
            self.latent_eval_loss = current_eval_loss # latent value to beat becomes current value if best
            self.eval_sucsincr = 0
            self.evalList = []
        else:
            self.eval_sucsincr += 1
            self.evalList.append(current_eval_loss)
        # train loss
        if current_train_loss < self.latent_train_loss:
            self.latent_train_loss = current_train_loss
            self.train_sucsincr = 0
        else:
            self.train_sucsincr += 1
        # delta
        current_delta_loss = current_train_loss - current_eval_loss
        if current_delta_loss <  self.latent_delta_loss:
            self.latent_delta_loss = current_delta_loss
            self.delta_sucsincr = 0 
            self.deltaList = []
        else:
            self.delta_sucsincr += 1
            self.deltaList.append(current_delta_loss)
            
        self.check_stopCiteria()
        
        return self.doStop, self.evalStop, self.evalList, self.deltaStop, self.deltaList
        
    def check_stopCiteria(self):
        if (self.eval_sucsincr > self.patience_eval):
            self.doStop = True
            self.evalStop = True
        
        if (self.delta_sucsincr > self.patience_dL):
            self.doStop = True
            self.deltaStop = True
        