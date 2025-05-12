
# Continous X Pipeline

![ResourcesProvisioned.png](../assets/ResourcesProvisioned.png)

## 1. Selecting Site

The Continous X Pipeline manages infrastructure mostly in KVM@TACC. So let's select the site!

from chi import server, context

context.version = "1.0"
context.choose_site(default="KVM@TACC")

Now that the Model Training, Model Evaluation, Model Serving and Data Pipeline are in place, we have to connect all the parts together. The main aim of this Continous_X_pipeline is to go from ideation to actual model deployment quickly and have an established process to iterate. This is indeed the Ops in the MLOps!

We will be provisioning resources and installing tools via code. For this we would be using :

-   Terraform: A declarative Infrastructure as Code (IaC) tool used to provision and manage cloud infrastructure (servers, networks, etc.) by defining the desired end state in configuration files. Here, we use it to provision our infrastructure.
-   Ansible: An imperative Configuration as Code (CaC) tool that automates system configuration, software installation, and application deployment through task-based YAML playbooks describing the steps to achieve a desired setup. Here, we use it to install Kubernetes and the Argo tools on our infrastructure after it is provisioned
-   Argo CD: A declarative GitOps continuous delivery tool for Kubernetes that automatically syncs and deploys applications based on the desired state stored in Git repositories.
-   Argo Workflows: A Kubernetes-native workflow engine where we define workflows, which execute tasks inside containers to run pipelines, jobs, or automation processes.

Let's get a copy of the Bird Classification Infrastructure repository

git clone --recurse-submodules https://github.com/exploring-curiosity/MLOps.git

## 2. Setup Environment


### Install and configure Terraform

Before we can use Terraform, we’ll need to download a Terraform client. The following cell will download the Terraform client and “install” it in this environment:

mkdir -p /work/.local/bin
wget https://releases.hashicorp.com/terraform/1.10.5/terraform_1.10.5_linux_amd64.zip
unzip -o -q terraform_1.10.5_linux_amd64.zip
mv terraform /work/.local/bin
rm terraform_1.10.5_linux_amd64.zip

The Terraform client has been installed to: /work/.local/bin. In order to run terraform commands, we will have to add this directory to our PATH, which tells the system where to look for executable files.

export PATH=/work/.local/bin:$PATH

Let’s make sure we can now run `terraform` commands. The following cell should print usage information for the `terraform` command, since we run it without any subcommands:

terraform

### Configure the PATH

Both Terraform and Ansible executables have been installed to a location that is not the system-wide location for executable files: `/work/.local/bin`. In order to run `terraform` or `ansible-playbook` commands, we will have to add this directory to our `PATH`, which tells the system where to look for executable files.

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

### Prepare Kubespray

To install Kubernetes, we’ll use Kubespray, which is a set of Ansible playbooks for deploying Kubernetes. We’ll also make sure we have its dependencies now:

PYTHONUSERBASE=/work/.local pip install --user -r MLOps/continous_X_pipeline/ansible/k8s/kubespray/requirements.txt

We will need to prepare credentials with which it can act on our behalf on the Chameleon OpenStack cloud. This is a one-time procedure.

To get credentials, open the Horizon GUI:

-   from the Chameleon website
-   click “Experiment” \> “KVM@TACC”
-   log in if prompted to do so
-   check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

On the left side, expand the “Identity” section and click on “Application Credentials”. Then, click “Create Application Credential”.

-   In the “Name”, field, use “mlops-lab”.
-   Set the “Expiration” date.
-   Click “Create Application Credential”.
-   Choose “Download clouds.yaml”.

Let's add the actual application_credential_id and application_credential_secret from the downloaded clouds.yaml in the clouds.yaml file that we see here. 

Terraform will look for the `clouds.yaml` in either `~/.config/openstack` or the directory from which we run `terraform` - we will move it to the latter directory:

cp clouds.yaml /work/gourmetgram-iac/tf/kvm/clouds.yaml

## 3. Provision infrastructure with Terraform

Now that everything is set up, we are ready to provision our VM resources with Terraform! We will use Terraform to provision 3 VM instances and associated network resources on the OpenStack cloud.

### Preliminaries
Let’s navigate to the directory with the Terraform configuration for our KVM deployment:

cd /work/MLOps/continous_X_pipeline/tf/kvm/

and make sure we’ll be able to run the terraform executable by adding the directory in which it is located to our PATH:

export PATH=/work/.local/bin:$PATH

We also need to un-set some OpenStack-related environment variables that are set automatically in the Chameleon Jupyter environment, since these will override some Terraform settings that we *don’t* want to override:

unset $(set | grep -o "^OS_[A-Za-z0-9_]*")

### Understanding our Terraform configuration
[data.tf](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/tf/kvm/data.tf) :  data sources gets existing infrastructure details from OpenStack about resources *not* managed by Terraform.

[main.tf](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/tf/kvm/main.tf) :
Here we actually allocate the resources. Except for the block storage, which is previously allocated by the data team member and we have to attach it to node1. We are attaching it to node1 because in our kubernetes cluster, node1 will be the leader node. We are doing this because we want to persist data beyond the lifecycle of our VM instances. This persistent block store will have the storage for MinIO, PostgreSQL, etc.

[variables.tf](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/tf/kvm/variables.tf) : lets us define inputs and reuse the configuration across different environments. The value of variables can be passed in the command line arguments when we run a `terraform` command, or by defining environment variables that start with `TF_VAR`. In this example, there’s a variable `instance_hostname` so that we can re-use this configuration to create a VM with any hostname - the variable is used inside the resource block with `name = "${var.instance_hostname}"`.

### Applying our Terraform configuration
First, we need Terraform to set up our working directory, make sure it has “provider” plugins to interact with our infrastructure provider (it will read in `provider.tf` to check), and set up storage for keeping track of the infrastructure state:

terraform init

To follow the project naming conventions and adding the key. Not this is the name of the key followed by the entire team members.

export TF_VAR_suffix=project38
export TF_VAR_key=id_rsa_chameleon_project_g38

We should confirm that our planned configuration is valid:

terraform validate

terraform apply -auto-approve

## 4. Ansible
Now that we have provisioned some infrastructure, we can configure and install software on it using Ansible!

### Preliminaries

As before, let’s make sure we’ll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

If you haven’t already, make sure to put your floating IP (which you can see in the output of the Terraform command!) in the `ansible.cfg` configuration file, and move it to the specified location.

The following cell will show the contents of this file, so you can double check - make sure your real floating IP is visible in this output!

cp ansible.cfg /work/MLOps/continous_X_pipeline/ansible/ansible.cfg

### Verify connectivity

First, we’ll run a simple task to check connectivity with all hosts listed in the [inventory.yaml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/inventory.yml)

    all:
      vars:
        ansible_python_interpreter: /usr/bin/python3
      hosts:
        node1:
          ansible_host: 192.168.1.11
          ansible_user: cc
        node2:
          ansible_host: 192.168.1.12
          ansible_user: cc
        node3:
          ansible_host: 192.168.1.13
          ansible_user: cc

It uses the `ping` module, which checks if Ansible can connect to each host via SSH and run Python code there.

But to be able to do that we would need to add the private key in .ssh folder. 

cp /work/id_rsa_chameleon_project_g38 /work/.ssh/

cd /work/.ssh

chmod 600 id_rsa_chameleon_project_g38

ssh-add id_rsa_chameleon_project_g38

cd /work/MLOps/continous_X_pipeline/ansible/

ansible -i inventory.yml all -m ping

### Run a “Hello, World” playbook

ansible-playbook -i inventory.yml general/hello_host.yml

This was just a sanity check!

## 5. Deploy Kubernetes using Ansible

### Preliminaries
As before, let’s make sure we’ll be able to use the Ansible executables. We need to put the install directory in the `PATH` inside each new Bash session.

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local

### Run a preliminary playbook

Before we set up Kubernetes, we will run a preliminary playbook to:

-   disable the host firewall on the nodes in the cluster.  We will also configure each node to permit the local container registry.
-   and, configure Docker to use the local registry.

Let's add the SSH key again for this new session

cd /work/.ssh/

ssh-add id_rsa_chameleon_project_g38

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml

### Run the Kubespray play

Then, we can run the Kubespray playbook!

export ANSIBLE_CONFIG=/work/MLOps/continous_X_pipeline/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles

cd /work/MLOps/continous_X_pipeline/ansible/k8s/kubespray

ansible-playbook -i ../inventory/mycluster --become --become-user=root ./cluster.yml

### Access the ArgoCD
In the local termial we will have to run :\
ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D

runs on node1 \
kubectl port-forward svc/argocd-server -n argocd 8888:443

https://127.0.0.1:8888/

Use the Username and Password from the above output

### Access the ArgoCD
In the local termial we will have to run :\
ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D

runs on node1 \
kubectl port-forward svc/argocd-server -n argocd 8888:443

https://127.0.0.1:8888/

Use the Username and Password from the above output

## 6. ArgoCD to manage applications on the Kubernetes cluster

With our Kubernetes cluster up and running, we are ready to deploy applications on it!

export PATH=/work/.local/bin:$PATH
export PYTHONUSERBASE=/work/.local
export ANSIBLE_CONFIG=/work/MLOps/continous_X_pipeline/ansible/ansible.cfg
export ANSIBLE_ROLES_PATH=roles

First, we will deploy our birdclef “platform”. This has all the “accessory” services we need to support our machine learning application.

Let’s add the birdclef-platform application now. In the output of the following cell, look for the MinIO secret, which will be generated and then printed in the output:

cd /work/.ssh

ssh-add id_rsa_chameleon_project_g38

cd /work/MLOps/continous_X_pipeline/ansible/

ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml

Let's analyse the code for [argocd_add_platform](https://github.com/exploring-curiosity/MLOps/edit/main/continous_X_pipeline/ansible/argocd/argocd_add_platform.yml) :

Here we are executing the following tasks.

1. Creating Directory for the mount : Since we are using a persistent Block Storage which already exists, we will have to run two commands. One to build the directory and the other to mount it.

2. We next mount the directory

Note : As of this writing, this code has been commented out we do not yet have the persistent Block Storage for the services. This would also require us to modify the files in Platform Helm charts, but has been left out for now to ensure the complete integration works.

3. We next get the ArgoCD admin password from Kubernetes secret

4. We Decode ArgoCD admin password

5. We Log in to ArgoCD

6. Add repository to ArgoCD. This helps in syncing the platform for any changes in the Kubernetes Manifest files.

7. We ensure birdclef-platform namespace exists

8. We create birdclef-platform namespace if missing

9. Check if MinIO secret already exists, in case we are running this flow again

10. If we are running this flow for the first time, we generate MinIO secret key.

11. Fetching existing MinIO secret key if already exists.

12. Decoding existing MinIO secret key

13. Check if ArgoCD application exists

14. Create ArgoCD Helm application (like MinIO, MLFLow, PostgreSQL, Prometheus, LabelStudio etc) if it does not exist.

15. Update ArgoCD Helm application if it exists

16. Display MinIO credentials to login.

After running this flow for the first time, any changes made in Helm Application via git will directly be reflected in ArgoCD. 

Once the platform is deployed, we can open:
(substitute A.B.C.D with floating IP) \
MinIO object Store :  http://A.B.C.D:9001 \
MLFlow             :  http://A.B.C.D:8000  \
Label-Studio : http://A.B.C.D:5000 \
Prometheus : http://A.B.C.D:4000 \
Grafana : http://A.B.C.D:3000 

Next, we need to deploy the Bird Classification application. Before we do, we need to build a container image. We will run a one-time workflow in Argo Workflows to build the initial container images for the “staging”, “canary”, and “production” environments:

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/workflow_build_init.yml

Through this workflow : [workflow_build_init](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/argocd/workflow_build_init.yml)

we are calling the [build-initial.yaml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/workflows/build-initial.yaml) file which executes the following tasks :

Builds the initial container images for staging, canary, and production using the [FastAPI wrapper](https://github.com/harishbalajib/BirdClassification) for the model. 

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml

By executing the workflow [argocd_add_staging.yml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/argocd/argocd_add_staging.yml) we are primarily creating the birdclef-staging namespace which we can monitor in ArgoCD. And by using this worflow, we are executing [staging](https://github.com/exploring-curiosity/MLOps/tree/main/continous_X_pipeline/k8s/staging) manifest, where we actually create a container for the staging environment from the above staging image we created.

At the end of this workflow, our application should be up and running and available at http://A.B.C.D:8081 (where A.B.C.D is our public IP)

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml

By executing the workflow [argocd_add_canary.yml](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/ansible/argocd/argocd_add_canary.yml) we are primarily creating the birdclef-canary namespace which we can monitor in ArgoCD. And by using this worflow, we are executing [canary](https://github.com/exploring-curiosity/MLOps/tree/main/continous_X_pipeline/k8s/canary) manifest, where we actually create a container for the canary environment from the above canary image we created.

At the end of this workflow, our application should be up and running and available at http://A.B.C.D:8080 (where A.B.C.D is our public IP)

cd /work/MLOps/continous_X_pipeline/ansible

ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml

By executing the workflow argocd_add_prod.yml we are primarily creating the birdclef-production namespace which we can monitor in ArgoCD. And by using this worflow, we are executing production manifest, where we actually create a container for the staging environment from the above production image we created.

At the end of this workflow, our application should be up and running and available at http://A.B.C.D (where A.B.C.D is our public IP)

cd /work/MLOps/continous_X_pipeline/ansible

Now, we will manage our application lifecycle with Argo Worfklows. We will understand these workflow more in depth in the next sections. 

ansible-playbook -i inventory.yml argocd/workflow_templates_apply.yml

## 7. Model and application lifecycle - Part 1

### Run a training and evaluation job

In this we will manually trigger a model training and evaluation. This manual trigger could be for several reasons like an update in the model or change in production data.

Through the previous workflow, we created a worflow template in Argo Workflow named [train-model](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/workflows/train-model.yaml), which is responsible for both training and evaluating the model.

Let's look at the code to understand the flow better

This template accepts 3 Public Addresses as shown here :
spec:
entrypoint: training-and-build
arguments:
parameters:
- name: train-ip
- name: eval-ip
- name: mlflow-ip

**train-ip** is the Public IP to trigger the Training Endpoint, which is responsible for Triggering the training process of the model, and logging the model artificats and status in ML Flow.

**eval-ip** is the Public IP address to trigger the Model Evaluation Endpoint. This endpoint is responsible for Evaluating the model and registering it in the MLFlow with a Specific Name.

**mlflow-ip** is the Public IP where the MLFlow is accessible. It will become more clear on why we are using the MLFlow here from the code below :

Through this workflow, a Training Endpoint is triggered with the help of **train-ip**. This is an API call. We get a RUN ID of the Model that is logged in the MLFlow. The following code achieves this :
```
RESPONSE=$(curl -f -s -X POST " http://{{inputs.parameters.train-ip}}:9090/train?model_name=resnet50&data_source=train")
        CURL_EXIT_CODE=$?
        echo "[INFO] Training endpoint response was: $RESPONSE" >&2
        if [ $CURL_EXIT_CODE -ne 0 ]; then
          echo "[ERROR] curl failed with code $CURL_EXIT_CODE" >&2
          exit $CURL_EXIT_CODE
        fi
        echo "[INFO] Training endpoint response was: $RESPONSE" >&2

```

Now, its possible that we Model Training could take several minutes if not hours togther to train a complex model on large datasets. And HTTP Endpoint calls have a timeout of just a few minutes.

Hence, the Model Training Endpoint immediately return the RUN_ID of the model which is logs in the MLFlow.

As Part of this workflow, we next keep polling the MLFlow to check if the process of training the model has completed. The following code achieves just this. We extract the RUN_ID and use it to poll the [MLFlow API](https://mlflow.org/docs/latest/api_reference/rest-api.html#get-run) to track the status.

```
 RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')    
        if [ -z "$RUN_ID" ]; then
          echo "[ERROR] run_id not found in response" >&2
          exit 1
        fi
        echo "[INFO] MLflow run ID: $RUN_ID" >&2
        
        #Polling MLFlow
        TERMINAL="FINISHED|FAILED|KILLED"
        while true; do
          STATUS=$(curl -s "http://{{inputs.parameters.mlflow-ip}}:8000/api/2.0/mlflow/runs/get?run_id=${RUN_ID}"| jq -r '.run.info.status')
          echo "[INFO] Run ${RUN_ID} status: ${STATUS}" >&2
          case "$STATUS" in
            FINISHED|FAILED|KILLED)
              echo "[INFO] Terminal state reached: $STATUS" >&2
              break
              ;;
          esac
          sleep 10
        done
```

Now that the model is ready, we next have to evaluate it and register it. Since this could also happen outside the current Kubernetes we have to trigger another Endpoint. At the end of this process we get the version of a registered model (named as 'BirdClassificationModel'). The following code demonstrates this:

```
EVAL_RESPONSE=$(curl -f -s -X GET "http://{{inputs.parameters.eval-ip}}:8080/get-version?run_id=${RUN_ID}")
        CURL_EXIT_CODE=$?
        echo "[INFO] Evaluation endpoint response was: EVAL_RESPONSE" >&2
        if [ $CURL_EXIT_CODE -ne 0 ]; then
          echo "[ERROR] curl failed with code $CURL_EXIT_CODE" >&2
          exit $CURL_EXIT_CODE
        fi
         
        # Extracting model version
        VERSION=$(echo "EVAL_RESPONSE" | jq -r '.new_model_version // empty')

        if [ -z "$VERSION" ]; then
          echo "[WARN] 'new_model_version' not found in response." >&2
          exit 1
        fi

        echo -n "$VERSION"
```

So, we triggered model train, model evaluation and at the end got the version of the registered model. With this the model's artifacts could be downloaded from MLFlow.

To actually trigger this template, we have to go to Argo Workflows > Workflow Templates > Submit > Add the train-ip, eval-ip and mlflow-ip > Hit Submit.

Now that we have a new registered model, we need a new container build!

This is triggered *automatically* when a new model version is returned from a training job.

One the build successful, and if we trigger http://A.B.C.D:8081 , we would be accessing the latest model which we just obtained.


This completes the critical flow of obtaining a model. So now we have our FastAPI wrapper for this model, replace the existing model with this new model (with the name bird.pth). The FastAPI just loads this model and now, this model is available to users!

Let's understand that flow better in the next section!

## 8. Model and application lifecycle - Part 2

Once we have a container image, the progression through the model/application lifecycle continues as the new version is promoted through different environments:

-   **Staging**: The container image is deployed in a staging environment that mimics the “production” service but without live users. In this staging environmenmt, we can perform integration tests against the service and also load tests to evaluate the inference performance of the system.
-   **Canary** (or blue/green, or other “preliminary” live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems.
-   **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.

### Promoting a Model

Now that we have tested our new model in staging, it time to promote it to canary. And from Canary to staging.

We can do this by following the below steps :

1. Workflow Templates > promote-model > Submit
2. In the source environment, mention "staging" and type "canary" in target-environment.
3. Select the desired model version in staging to be promoted. This could be obtained from MLFlow.
4. Hit Submit

After this [build-container-image](https://github.com/exploring-curiosity/MLOps/blob/main/continous_X_pipeline/workflows/build-container-image.yaml) will be triggered automatically, which downloads the code for the model wrapper from git, downloads the staging model from MLFlow, bundles both of them together and makes it avialable in the canary environment.

So now if we go to http://A.B.C.D/8080 we would be accessing the latest model from the canary environment.

We can follow the same approach to promote the model from canary to production. 


## 9. Delete infrastructure with Terraform

Since we provisioned our infrastructure with Terraform, we can also delete all the associated resources using Terraform.

cd /work/MLOps/continous_X_pipeline/tf/kvm

export PATH=/work/.local/bin:$PATH

unset $(set | grep -o "^OS_[A-Za-z0-9_]*")

export TF_VAR_suffix=project_38
export TF_VAR_key=id_rsa_chameleon_project_g38








